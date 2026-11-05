# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMX-only, high-performance MLA backend for DeepSeek V2/V3/R1 on CPU.

Built on the AMX decode/extend/bmm kernels vendored under
``csrc/cpu/sgl-kernels/``, plugged into vLLM's ``MLACommonBackend``/
``MLACommonImpl`` abstraction the same way every other concrete MLA backend
(``TritonMLAImpl``, etc.) does.

This is a separate backend from the reference ``CPUMLABackend``
(``vllm/v1/attention/backends/mla/cpu_mla.py``): that one targets every CPU
(any dtype, block_size=16, ``mla_decode_kvcache`` decode kernel + inherited
SDPA-style prefill) as a functional/CI reference, not performance. This
backend instead requires AMX (bf16 only, block_size a multiple of 32) and is
selected by the platform layer in preference to the reference backend
whenever the host supports it -- see ``CpuPlatform.get_attn_backend_cls``.

Two points where this backend differs structurally from the GPU backends,
both explained in the CPU MLA design plan:

- ``forward_mha`` is fully overridden (not inherited from
  ``MLACommonBaseImpl``): the GPU "compute-friendly" prefill path depends on
  a pluggable ``MLAPrefillBackend`` and CUDA-only chunked-context gather ops,
  neither of which exist on CPU. Instead, this attends directly in
  latent-MQA space via the ``extend_attention_cpu`` kernel, which handles
  cached-prefix continuation and fresh prefill in one causal pass (the KV
  cache already contains the new tokens by the time this runs, since
  ``do_kv_cache_update`` executes first).
- Weight absorption for the prefill path is done with this impl's own
  VNNI-packed copies of ``W_UK``/``W_UV`` (computed in
  ``process_weights_after_loading``, using ``bmm_cpu``), rather than reading
  the generic ``layer.W_UK_T``/``layer.W_UV``/``layer._v_up_proj`` the way
  GPU backends do, since ``forward_mha``'s abstract signature has no
  ``layer`` parameter (only ``forward_mqa`` does) and that signature is left
  untouched. The decode path needs no such packing: ``MLAAttention.forward_impl``
  already absorbs/de-absorbs Q around the ``forward_mqa`` call using the
  generic (unpacked) ``layer.W_UK_T``/``layer._v_up_proj``, so
  ``forward_mqa`` here only has to invoke the decode kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.cpu import W8A16_MAX_TOKENS
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf

_MIN_WORK_PER_SPLIT = 32
_SPLIT_OCCUPANCY_MULTIPLIER = 8
_MAX_KV_SPLITS = 64


def _block_quant_to_tensor_quant(
    weight: torch.Tensor,  # FP8 [N, K] block-quantized weight
    weight_scale: torch.Tensor,  # [ceil(N/bn), ceil(K/bk)] block scales
    block_size: list[int],  # [block_n, block_k]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize FP8 block-quant weight to FP8 per-tensor quant.

    Returns (fp8_weight, dequant_scale) where fp8_weight * dequant_scale
    approximates the original float weight.  dequant_scale is a 0-dim
    scalar tensor as expected by bmm_cpu's ``scale`` argument.

    NOTE: block->tensor re-quantization may slightly affect accuracy
    (equivalent to SGLang's block_quant_to_tensor_quant for CPU).
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = weight.shape

    # Expand block scales [ceil(N/bn), ceil(K/bk)] -> [N, K]
    scale = weight_scale.to(torch.float32)
    if scale.shape[0] < n:
        scale = scale.repeat_interleave(block_n, dim=0)[:n, :]
    if scale.shape[1] < k:
        scale = scale.repeat_interleave(block_k, dim=1)[:, :k]

    # Dequantize to float32
    x_f32 = weight.to(torch.float32) * scale

    # Re-quantize per-tensor back to FP8
    fp8_max = torch.finfo(weight.dtype).max  # 448.0 for e4m3fn
    amax = x_f32.abs().amax().clamp(min=1e-12)
    quant_scale = fp8_max / amax
    x_fp8 = (x_f32 * quant_scale).clamp(-fp8_max, fp8_max).to(weight.dtype)

    # dequant_scale = 1 / quant_scale (0-dim tensor for bmm_cpu)
    dequant_scale = (amax / fp8_max).to(torch.float32)
    return x_fp8, dequant_scale


def _compute_num_kv_splits(
    num_reqs: int, num_heads: int, max_seq_len: int, num_threads: int
) -> int:
    """Split the KV length until (request, head, split) work items occupy the
    CPU threads; long sequences cap the split count at one per
    ``_MIN_WORK_PER_SPLIT`` tokens."""
    target = num_threads * _SPLIT_OCCUPANCY_MULTIPLIER
    wanted = -(-target // max(1, num_reqs * num_heads))
    splits = 1
    while splits < min(wanted, _MAX_KV_SPLITS):
        splits *= 2
    return max(1, min(splits, max_seq_len // _MIN_WORK_PER_SPLIT))


def _expand_block_table(block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Adapter: vLLM's block-table paging -> the flat per-(request, position)
    physical row index the decode/extend kernels expect. Called once per step
    from ``AMXMLAMetadataBuilder.build``, not per layer.
    """
    offsets = torch.arange(
        block_size, device=block_table.device, dtype=block_table.dtype
    )
    flat = block_table.unsqueeze(-1) * block_size + offsets
    return flat.reshape(block_table.size(0), -1).contiguous()


class AMXMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank(512) + qk_rope_head_dim(64), DeepSeek V2/V3/R1/V3.2.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 32 == 0

    @staticmethod
    def get_name() -> str:
        return "AMX_MLA"

    @staticmethod
    def get_impl_cls() -> type[AMXMLAImpl]:
        return AMXMLAImpl

    @staticmethod
    def get_builder_cls() -> type[AMXMLAMetadataBuilder]:
        return AMXMLAMetadataBuilder


@dataclass
class AMXMLADirectMetadata:
    """Per-step inputs of `AMXMLADirect`, shared by the layers of one group.

    Fixed-capacity buffers refreshed in place at every build, so the compiled
    graph never sees a shape change. Decode requests occupy the first rows of
    `req_to_token`, prefill requests the rows after them.
    """

    # [decode tokens, prefill tokens, prefill requests, decode kv splits]
    scalars: torch.Tensor
    slot_mapping: torch.Tensor
    req_to_token: torch.Tensor
    dec_req_pool_indices: torch.Tensor
    dec_seq_lens: torch.Tensor
    pre_req_pool_indices: torch.Tensor
    pre_seq_lens: torch.Tensor
    pre_extend_seq_lens: torch.Tensor
    pre_extend_start_loc: torch.Tensor

    @classmethod
    def allocate(cls, vllm_config, block_size: int) -> AMXMLADirectMetadata:
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_context = -(-vllm_config.model_config.max_model_len // block_size)
        max_context *= block_size
        i64 = {"dtype": torch.int64}
        i32 = {"dtype": torch.int32}
        return cls(
            scalars=torch.zeros(4, **i64),
            slot_mapping=torch.full((max_tokens,), -1, **i64),
            # Only the rows and columns in use are read, so leave it unwritten.
            req_to_token=torch.empty(max_reqs, max_context, **i32),
            dec_req_pool_indices=torch.arange(max_reqs, **i64),
            dec_seq_lens=torch.zeros(max_reqs, **i64),
            pre_req_pool_indices=torch.zeros(max_reqs, **i64),
            pre_seq_lens=torch.zeros(max_reqs, **i64),
            pre_extend_seq_lens=torch.zeros(max_reqs, **i32),
            pre_extend_start_loc=torch.zeros(max_reqs, **i32),
        )

    def refresh(self, attn_metadata: MLACommonMetadata, slot_mapping: torch.Tensor):
        """Copies the step's metadata, built by `AMXMLAMetadataBuilder`, in."""
        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_tokens = attn_metadata.num_actual_tokens
        assert attn_metadata.num_decode_tokens == num_decodes, (
            "Direct MLA does not support multi-token decode (speculative decoding)."
        )
        assert num_decodes + num_prefills <= self.req_to_token.size(0)
        kv_splits = 1
        if attn_metadata.decode is not None:
            decode = attn_metadata.decode
            rows = decode.req_to_token  # type: ignore[attr-defined]
            self.req_to_token[:num_decodes, : rows.size(1)].copy_(rows)
            self.dec_seq_lens[:num_decodes].copy_(decode.seq_lens_i64)  # type: ignore[attr-defined]
            kv_splits = decode.num_kv_splits  # type: ignore[attr-defined]
        if attn_metadata.prefill is not None:
            prefill = attn_metadata.prefill
            rows = prefill.req_to_token  # type: ignore[attr-defined]
            end = num_decodes + num_prefills
            self.req_to_token[num_decodes:end, : rows.size(1)].copy_(rows)
            self.pre_req_pool_indices[:num_prefills].copy_(
                torch.arange(num_decodes, end)
            )
            self.pre_seq_lens[:num_prefills].copy_(prefill.cpu_seq_lens)  # type: ignore[attr-defined]
            self.pre_extend_seq_lens[:num_prefills].copy_(prefill.extend_seq_lens)  # type: ignore[attr-defined]
            self.pre_extend_start_loc[:num_prefills].copy_(prefill.extend_start_loc)  # type: ignore[attr-defined]
        self.slot_mapping[:num_tokens].copy_(slot_mapping[:num_tokens])
        self.scalars.copy_(
            torch.tensor(
                [num_decodes, num_tokens - num_decodes, num_prefills, kv_splits]
            )
        )


def init_direct_mla_layers(layers, vllm_config) -> None:
    """Gives the direct MLA layers their shared metadata before the model is
    traced (it is traced before the KV cache and the builders exist). The zero
    scalars make the dummy run skip the layers."""
    shared = None
    for layer in layers:
        if isinstance(layer, MLAAttention) and getattr(
            layer.impl, "use_cpu_direct", False
        ):
            if shared is None:
                shared = AMXMLADirectMetadata.allocate(
                    vllm_config, vllm_config.cache_config.block_size
                )
            layer.cpu_mla_metadata = shared


class AMXMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    def __init__(
        self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs
        )
        # Layers called directly from compiled code (`AMXMLADirect`) share one
        # metadata instance that is refreshed in place at every build.
        self._direct: AMXMLADirectMetadata | None = None
        if envs.VLLM_CPU_ATTN_DIRECT:
            layers = get_layers_from_vllm_config(vllm_config, MLAAttention, layer_names)
            # The graph was traced against the instance `init_direct_mla_layers`
            # attached, so keep it.
            for layer in layers.values():
                self._direct = getattr(layer, "cpu_mla_metadata", self._direct)
            if self._direct is None:
                self._direct = AMXMLADirectMetadata.allocate(
                    vllm_config, kv_cache_spec.block_size
                )
            for layer in layers.values():
                layer.cpu_mla_metadata = self._direct

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        block_size = self.kv_cache_spec.block_size
        if attn_metadata.decode is not None:
            # Built once per step and reused by every layer, instead of
            # recomputing the same tensor arithmetic in every forward_mqa call.
            decode = attn_metadata.decode
            decode.req_to_token = _expand_block_table(  # type: ignore[attr-defined]
                decode.block_table, block_size
            )
            decode.seq_lens_i64 = decode.seq_lens.to(torch.int64)  # type: ignore[attr-defined]
            decode.req_pool_indices = torch.arange(  # type: ignore[attr-defined]
                decode.block_table.size(0),
                dtype=torch.int64,
                device=decode.block_table.device,
            )
            decode.num_kv_splits = _compute_num_kv_splits(  # type: ignore[attr-defined]
                decode.block_table.size(0),
                self.num_heads,
                attn_metadata.max_seq_len,
                current_platform.num_compute_units(),
            )
        if attn_metadata.prefill is not None:
            # cpu_seq_lens: total per-request context length (prefix + new
            # tokens), the one thing the extend kernel needs that isn't
            # already on the shared MLACommonPrefillMetadata.
            prefill = attn_metadata.prefill
            num_decodes = attn_metadata.num_decodes
            num_prefills = attn_metadata.num_prefills
            prefill.cpu_seq_lens = common_attn_metadata.seq_lens[  # type: ignore[attr-defined]
                num_decodes : num_decodes + num_prefills
            ].to(torch.int64)
            prefill.req_to_token = _expand_block_table(  # type: ignore[attr-defined]
                prefill.block_table, block_size
            ).to(torch.int64)
            prefill.req_pool_indices = torch.arange(  # type: ignore[attr-defined]
                prefill.block_table.size(0),
                dtype=torch.int64,
                device=prefill.block_table.device,
            )
            query_start_loc_i64 = prefill.query_start_loc.to(torch.int64)
            extend_seq_lens = query_start_loc_i64[1:] - query_start_loc_i64[:-1]
            prefill.extend_seq_lens = extend_seq_lens  # type: ignore[attr-defined]
            prefill.extend_start_loc = query_start_loc_i64[:-1]  # type: ignore[attr-defined]
            prefill.max_len_extend = int(extend_seq_lens.max().item())  # type: ignore[attr-defined]
        if self._direct is not None:
            self._direct.refresh(attn_metadata, common_attn_metadata.slot_mapping)
        return attn_metadata


class AMXMLAImpl(MLACommonImpl[MLACommonMetadata]):
    # Tells MLAAttention (mla_attention.py) to use this impl's own packed
    # W_UK/W_UV + bmm_cpu for the shared decode absorb/de-absorb, mirroring
    # is_aiter_triton_fp8_bmm_enabled/is_aiter_triton_fp4_bmm_enabled.
    uses_amx_bmm = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "AMXMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for AMXMLAImpl"
            )
        self.use_cpu_direct = envs.VLLM_CPU_ATTN_DIRECT
        self._w_uk_packed: torch.Tensor | None = None
        self._w_uv_packed: torch.Tensor | None = None
        self._w_scale: torch.Tensor | None = (
            None  # per-tensor dequant scale (FP8 W8A16)
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # forward_mha (unlike forward_mqa) receives no `layer` argument, so it
        # cannot read the generic layer.W_UK_T/W_UV/._v_up_proj that
        # MLAAttention.process_weights_after_loading already computes. This
        # impl derives its own copies from self.kv_b_proj (already available
        # via the constructor) and VNNI-packs them once, ahead of time, for
        # use with bmm_cpu in forward_mha.
        #
        # kv_b_proj carries `_cpu_skip_gemm_dispatch`, so the CPU FP8 kernels
        # leave its weight in the checkpoint [N, K] layout. For block-quant
        # FP8 checkpoints, re-quantize per-tensor and keep the BMM in FP8
        # (W8A16); everything else goes through the BF16 path.
        weight = getattr(self.kv_b_proj, "weight", None)
        block_size = getattr(self.kv_b_proj, "weight_block_size", None)
        if (
            weight is not None
            and weight.dtype == torch.float8_e4m3fn
            and block_size is not None
        ):
            block_scale = getattr(self.kv_b_proj, "weight_scale_inv", None)
            if block_scale is None:
                block_scale = self.kv_b_proj.weight_scale
            fp8_weight, self._w_scale = _block_quant_to_tensor_quant(
                weight, block_scale, block_size
            )
            # [N, K] -> [K, N] = [kv_lora_rank, heads * (nope + v)]
            kv_b_proj_weight = fp8_weight.T.contiguous().view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
        else:
            kv_b_proj_weight = get_and_maybe_dequant_weights(
                self.kv_b_proj, out_dtype=act_dtype
            ).T
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            self._w_scale = None
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # bmm_cpu computes out[b] = mat1[b] @ mat2[b]^T (Linear-weight
        # convention: mat2[b] is (OUT, IN)).
        #   absorb:    ql_nope(N,B,L) = q_nope(N,B,P) @ w_uk_for_bmm(N,L,P)^T
        #   de-absorb: out(N,B,V)     = attn_out(N,B,L) @ w_uv_for_bmm(N,V,L)^T
        w_uk_for_bmm = w_uk.permute(1, 0, 2).contiguous()  # (N, L, P)
        w_uv_for_bmm = w_uv.permute(1, 2, 0).contiguous()  # (N, V, L)
        self._w_uk_packed = torch.ops._C.convert_weight_packed(w_uk_for_bmm)
        self._w_uv_packed = torch.ops._C.convert_weight_packed(w_uv_for_bmm)

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        skip_scale: bool = False,
    ) -> None:
        # Overrides the default (CUDA-only concat_and_cache_mla).
        if kv_cache.numel() == 0:
            return
        assert kv_cache_dtype == "auto", (
            "AMXMLAImpl only supports an unquantized (bf16) KV cache; fp8 "
            "KV cache for CPU MLA is not yet implemented."
        )
        ops.amx_mla_concat_and_cache(
            kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten()
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives already absorbed (tensor or (ql_nope, q_pe) tuple) --
        # MLAAttention.forward_impl does the Q-absorption bmm generically
        # before calling forward_mqa, using layer.W_UK_T. De-absorption is
        # likewise done generically afterwards via layer._v_up_proj. Nothing
        # left for this impl to do beyond invoking the decode kernel.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        return self._decode_kernel(q, kv_c_and_k_pe_cache, attn_metadata), None

    def forward_decode_amx(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output: torch.Tensor,
    ) -> None:
        """MQA decode fused into one call: Q absorption, the decode kernel and
        V de-absorption, without the generic path's intermediate copies.

        Args:
            q: [B, N, qk_nope + qk_rope] query of the decode tokens.
            kv_c_and_k_pe_cache: The layer's KV cache.
            attn_metadata: Attention metadata with the decode fields set.
            output: [B, N * v_head_dim] slice of the layer output.

        """
        num_tokens, num_heads, _ = q.shape
        nope = self.qk_nope_head_dim
        rank = self.kv_lora_rank
        mqa_q = q.new_empty(num_tokens, num_heads, rank + self.qk_rope_head_dim)
        # bmm_cpu: out[n] = mat1[n] @ mat2[n]^T with (N, B, .) views of the
        # token-major tensors.
        ops.bmm_cpu(
            mqa_q[..., :rank].transpose(0, 1),
            q[..., :nope].transpose(0, 1),
            self._w_uk_packed,  # type: ignore[arg-type]
            True,
            self._w_scale,
        )
        mqa_q[..., rank:].copy_(q[..., nope:])
        attn_out = self._decode_kernel(mqa_q, kv_c_and_k_pe_cache, attn_metadata)
        ops.bmm_cpu(
            output.view(num_tokens, num_heads, self.v_head_dim).transpose(0, 1),
            attn_out.transpose(0, 1),
            self._w_uv_packed,  # type: ignore[arg-type]
            True,
            self._w_scale,
        )

    def _decode_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.decode is not None

        num_tokens = q.shape[0]
        kv_cache_flat = kv_c_and_k_pe_cache.view(-1, 1, self.head_size)
        v_buffer = kv_cache_flat[..., : self.kv_lora_rank]

        seq_lens = attn_metadata.decode.seq_lens_i64  # type: ignore[attr-defined]
        req_to_token = attn_metadata.decode.req_to_token  # type: ignore[attr-defined]
        req_pool_indices = attn_metadata.decode.req_pool_indices  # type: ignore[attr-defined]

        num_kv_splits = attn_metadata.decode.num_kv_splits  # type: ignore[attr-defined]
        # The kernel writes every element of both buffers before reading it.
        o = torch.empty(
            num_tokens,
            self.num_heads,
            self.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits = torch.empty(
            num_tokens,
            self.num_heads,
            num_kv_splits,
            self.kv_lora_rank + 1,
            dtype=torch.float32,
            device=q.device,
        )

        ops.cpu_mla_decode(
            q,
            kv_cache_flat,
            v_buffer,
            o,
            None,
            None,
            None,  # loc: only meaningful when key/value are given; we
            # already wrote the new K/V via do_kv_cache_update before this.
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            self.scale,
            0.0,
            False,
            0,
            None,
            None,
        )
        return o

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        assert output_scale is None, (
            "AMXMLAImpl.forward_mha does not support fused output quantization"
        )
        prefill = attn_metadata.prefill
        assert prefill is not None
        assert self._w_uk_packed is not None and self._w_uv_packed is not None

        num_tokens = q.shape[0]
        num_heads = self.num_heads

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Absorb: (N, B, P) x (N, L, P)^T -> (N, B, L)
        # bmm_cpu tolerates non-contiguous mat1/out as long as their last dim
        # is contiguous, so q_nope's transpose and ql_nope's transposed view
        # can be passed directly -- no copy needed for either.
        q_nope_t = q_nope.transpose(0, 1)
        ql_nope = torch.empty(
            num_tokens, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        ops.bmm_cpu(
            ql_nope.transpose(0, 1), q_nope_t, self._w_uk_packed, True, self._w_scale
        )
        mqa_q = torch.cat([ql_nope, q_pe], dim=-1)

        kv_cache_flat = kv_c_and_k_pe_cache.view(-1, 1, self.head_size)
        v_buffer = kv_cache_flat[..., : self.kv_lora_rank]

        req_to_token = prefill.req_to_token  # type: ignore[attr-defined]
        req_pool_indices = prefill.req_pool_indices  # type: ignore[attr-defined]

        seq_lens = prefill.cpu_seq_lens  # type: ignore[attr-defined]
        extend_seq_lens = prefill.extend_seq_lens  # type: ignore[attr-defined]
        extend_start_loc = prefill.extend_start_loc  # type: ignore[attr-defined]
        max_len_extend = prefill.max_len_extend  # type: ignore[attr-defined]

        # k_extend/v_extend: the new tokens' own latent K/V, aliased in one
        # 576-wide buffer (v_extend is a view of k_extend's first
        # kv_lora_rank columns) so the kernel can gather both from one read,
        # mirroring the cache's own layout.
        k_extend = torch.cat(
            [kv_c_normed, k_pe.reshape(num_tokens, self.qk_rope_head_dim)], dim=-1
        ).unsqueeze(1)
        v_extend = k_extend[..., : self.kv_lora_rank]

        attn_out = torch.empty(
            num_tokens, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        ops.cpu_mla_extend(
            mqa_q,
            k_extend,
            v_extend,
            attn_out,
            kv_cache_flat,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            max_len_extend,
            self.scale,
            0.0,
            False,
            0,
            None,
            None,
            None,
        )

        # De-absorb directly into `output` (view, no extra copy on either side):
        # (N, B, L) x (N, V, L)^T -> (N, B, V)
        attn_out_t = attn_out.transpose(0, 1)
        output_view = output.view(num_tokens, num_heads, self.v_head_dim).transpose(
            0, 1
        )
        ops.bmm_cpu(output_view, attn_out_t, self._w_uv_packed, True, self._w_scale)


class AMXMLADirect:
    """A whole MLA layer, decode and prefill tokens alike, in one native op.

    Reads its per-step metadata from `layer.cpu_mla_metadata` and the KV cache
    from the layer, so torch.compile traces it without an opaque custom op and
    Inductor can call the graph from C++ (`VLLM_CPU_ATTN_DIRECT`). Needs the
    FP8 block-quantized projections: the W8A16 copies that `_cpu_keep_w8a16`
    layers keep serve few-token steps, the W8A8 weights everything else.
    """

    def __init__(self, wrapper) -> None:
        impl = wrapper.mla_attn.impl
        self.layer = wrapper.mla_attn
        self.impl = impl
        qkv_a, q_b = wrapper.fused_qkv_a_proj, wrapper.q_b_proj
        self.weights = (
            qkv_a.w8a16_weight,
            qkv_a.w8a16_scale,
            q_b.w8a16_weight,
            q_b.w8a16_scale,
            qkv_a.weight,
            qkv_a.weight_scale_inv,
            q_b.weight,
            q_b.weight_scale_inv,
            impl._w_uk_packed,
            impl._w_uv_packed,
            wrapper.q_a_layernorm.weight.data,
            wrapper.kv_a_layernorm.weight.data,
            wrapper.rotary_emb.cos_sin_cache.to(
                wrapper.q_a_layernorm.weight.dtype
            ).contiguous(),
        )
        self.eps = wrapper.q_a_layernorm.variance_epsilon
        self.q_lora_rank = wrapper.q_lora_rank

    @staticmethod
    def create(wrapper) -> AMXMLADirect | None:
        impl = wrapper.mla_attn.impl
        if (
            not isinstance(impl, AMXMLAImpl)
            or not impl.use_cpu_direct
            or impl._w_uk_packed is None
        ):
            return None
        if (
            wrapper.q_lora_rank is None
            or wrapper.rotary_emb is None
            or wrapper.indexer is not None
            or wrapper.g_proj is not None
            or wrapper.dcp_q_replicate
        ):
            return None
        for proj in (wrapper.fused_qkv_a_proj, wrapper.q_b_proj):
            if (
                getattr(proj, "w8a16_weight", None) is None
                or getattr(proj, "weight_scale_inv", None) is None
                or proj.weight.dim() != 4
            ):
                return None
        return AMXMLADirect(wrapper)

    def __call__(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, out: torch.Tensor
    ) -> None:
        metadata = self.layer.cpu_mla_metadata
        impl = self.impl
        (
            qkv_a_w8a16,
            qkv_a_s8a16,
            q_b_w8a16,
            q_b_s8a16,
            qkv_a_w8a8,
            qkv_a_s8a8,
            q_b_w8a8,
            q_b_s8a8,
            w_kc,
            w_vc,
            q_ln,
            kv_ln,
            cos_sin_cache,
        ) = self.weights
        ops.mla_layer_cpu(
            hidden_states,
            positions,
            out,
            self.layer.kv_cache,
            qkv_a_w8a16,
            qkv_a_s8a16,
            q_b_w8a16,
            q_b_s8a16,
            qkv_a_w8a8,
            qkv_a_s8a8,
            q_b_w8a8,
            q_b_s8a8,
            w_kc,
            w_vc,
            q_ln,
            kv_ln,
            cos_sin_cache,
            impl._w_scale,
            metadata.scalars,
            metadata.slot_mapping,
            metadata.req_to_token,
            metadata.dec_req_pool_indices,
            metadata.dec_seq_lens,
            metadata.req_to_token,
            metadata.pre_req_pool_indices,
            metadata.pre_seq_lens,
            metadata.pre_extend_seq_lens,
            metadata.pre_extend_start_loc,
            self.eps,
            self.q_lora_rank,
            impl.kv_lora_rank,
            impl.qk_rope_head_dim,
            impl.scale,
            W8A16_MAX_TOKENS,
        )


class AMXMLAFusedDecode:
    """Decode-only MLA front end to back end in one native op.

    Replaces the qkv projections, the layernorms, rope, the KV-cache write, the
    decode kernel and the W_UV de-absorption of a pure-decode batch by a single
    call. Weights are the FP8 W8A16 copies that `_cpu_keep_w8a16` layers keep.
    """

    def __init__(self, wrapper) -> None:
        impl = wrapper.mla_attn.impl
        self.wrapper = wrapper
        self.impl = impl
        self.qkv_a_weight = wrapper.fused_qkv_a_proj.w8a16_weight
        self.qkv_a_scale = wrapper.fused_qkv_a_proj.w8a16_scale
        self.q_b_weight = wrapper.q_b_proj.w8a16_weight
        self.q_b_scale = wrapper.q_b_proj.w8a16_scale
        self.q_ln_weight = wrapper.q_a_layernorm.weight.data
        self.kv_ln_weight = wrapper.kv_a_layernorm.weight.data
        self.eps = wrapper.q_a_layernorm.variance_epsilon
        self.cos_sin_cache = wrapper.rotary_emb.cos_sin_cache.to(
            self.q_ln_weight.dtype
        ).contiguous()
        self.block_size = [128, 128]

    @staticmethod
    def create(wrapper) -> AMXMLAFusedDecode | None:
        impl = wrapper.mla_attn.impl
        if not isinstance(impl, AMXMLAImpl) or impl._w_uk_packed is None:
            return None
        if (
            wrapper.q_lora_rank is None
            or wrapper.rotary_emb is None
            or wrapper.indexer is not None
            or wrapper.g_proj is not None
            or wrapper.dcp_q_replicate
        ):
            return None
        for proj in (wrapper.fused_qkv_a_proj, wrapper.q_b_proj):
            if getattr(proj, "w8a16_weight", None) is None:
                return None
        return AMXMLAFusedDecode(wrapper)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        attn_metadata: MLACommonMetadata | None,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> bool:
        """Runs the fused op; returns False when the batch is not decode-only."""
        if (
            attn_metadata is None
            or attn_metadata.decode is None
            or attn_metadata.num_prefills > 0
            or slot_mapping is None
        ):
            return False
        num_tokens = hidden_states.shape[0]
        if attn_metadata.num_decode_tokens != num_tokens:
            return False
        decode = attn_metadata.decode
        impl = self.impl
        ops.mla_decode_fused_cpu(
            hidden_states,
            self.qkv_a_weight,
            self.q_b_weight,
            impl._w_uk_packed,  # type: ignore[arg-type]
            impl._w_uv_packed,  # type: ignore[arg-type]
            self.q_ln_weight,
            self.kv_ln_weight,
            positions[:num_tokens],
            self.cos_sin_cache,
            self.eps,
            self.qkv_a_scale,
            self.q_b_scale,
            impl._w_scale,
            self.block_size,
            self.wrapper.q_lora_rank,
            impl.kv_lora_rank,
            impl.qk_rope_head_dim,
            kv_cache,
            slot_mapping[:num_tokens],
            decode.req_to_token,  # type: ignore[attr-defined]
            decode.req_pool_indices,  # type: ignore[attr-defined]
            decode.seq_lens_i64,  # type: ignore[attr-defined]
            decode.num_kv_splits,  # type: ignore[attr-defined]
            impl.scale,
            out[:num_tokens],
        )
        return True


def mla_cpu_fused_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, mla_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    wrapper = mla_layer._fused_owner
    if not wrapper._cpu_fused(
        hidden_states, positions, output, attn_metadata, kv_cache, slot_mapping
    ):
        output.copy_(wrapper.forward_attn(positions, hidden_states))


def mla_cpu_fused_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return None


direct_register_custom_op(
    op_name="mla_cpu_fused_attention",
    op_func=mla_cpu_fused_attention,
    mutates_args=["output"],
    fake_impl=mla_cpu_fused_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)
