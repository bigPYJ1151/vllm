# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the AMX-only CPU MLA backend: the vendored
decode/extend/bmm kernels, the KV cache write, and the ``AMXMLAImpl``
backend built on top of them.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not torch.cpu._is_amx_tile_supported():
    pytest.skip("AMX MLA requires an AMX-capable host", allow_module_level=True)

torch.cpu._init_amx()

from vllm import _custom_ops as ops  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.attention.backends.mla.amx_mla import (  # noqa: E402
    AMXMLAImpl,
    _compute_num_kv_splits,
    _expand_block_table,
)

KV_LORA_RANK = 64
QK_ROPE_HEAD_DIM = 32
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 96, kv-cache row width
# The extend/decode kernels transpose by 512-bit lanes: both the cache row
# width and the (kv_lora_rank-wide) value width must be 32-element aligned.
DTYPE = torch.bfloat16
ATOL = 2e-2
RTOL = 2e-2
# Backend-level tests chain absorb -> attend -> de-absorb through bf16,
# accumulating more rounding error than a single kernel op.
_IMPL_ATOL = 1.5e-1
_IMPL_RTOL = 1.5e-1


def _flatten_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """(num_blocks, block_size, head_size) -> (num_slots, head_size)."""
    return kv_cache.view(-1, kv_cache.size(-1))


def _random_paged_cache(num_slots: int, head_size: int) -> torch.Tensor:
    return torch.randn(num_slots, head_size, dtype=DTYPE)


def _ref_latent_attn(
    q: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Plain-PyTorch causal-free latent-space MQA attention for one request.

    q: (num_heads, head_size); keys: (ctx_len, head_size). Values are the
    first ``kv_lora_rank`` columns of the same latent rows (MLA's K/V
    aliasing), matching what decode_attention_cpu/extend_attention_cpu
    compute internally for the MLA-shaped case.
    """
    values = keys[:, :kv_lora_rank]
    scores = (q.float() @ keys.float().T) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ values.float()).to(q.dtype)


def test_bmm_cpu_matches_torch_bmm():
    set_random_seed(0)
    # bmm_cpu requires the output's last dim (mat2's out-features) to be a
    # multiple of 32 (tinygemm's tile width); the contraction dim is free.
    n, b, p, l = 8, 32, 32, 32  # noqa: E741
    mat1 = torch.randn(n, b, p, dtype=DTYPE)
    mat2 = torch.randn(n, l, p, dtype=DTYPE)  # (N, OUT, IN) Linear convention

    ref = torch.bmm(mat1.float(), mat2.float().transpose(1, 2)).to(DTYPE)

    out = torch.empty(n, b, l, dtype=DTYPE)
    ops.bmm_cpu(out, mat1, mat2, False, None)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    packed = torch.ops._C.convert_weight_packed(mat2.contiguous())
    out_vnni = torch.empty(n, b, l, dtype=DTYPE)
    ops.bmm_cpu(out_vnni, mat1, packed, True, None)
    torch.testing.assert_close(out_vnni, ref, atol=ATOL, rtol=RTOL)


def test_concat_and_cache_mla_round_trip():
    set_random_seed(0)
    num_tokens = 37
    num_blocks, block_size = 8, 16
    kv_cache = torch.zeros(num_blocks, block_size, HEAD_SIZE, dtype=DTYPE)

    kv_c_normed = torch.randn(num_tokens, KV_LORA_RANK, dtype=DTYPE)
    k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    slot_mapping = torch.randperm(num_blocks * block_size)[:num_tokens].to(torch.int64)

    ops.amx_mla_concat_and_cache(kv_c_normed, k_pe, kv_cache, slot_mapping)

    flat = _flatten_cache(kv_cache)
    read_back = flat[slot_mapping]
    expected = torch.cat([kv_c_normed, k_pe], dim=-1)
    torch.testing.assert_close(read_back, expected)

    untouched = torch.ones(num_blocks * block_size, dtype=torch.bool)
    untouched[slot_mapping] = False
    assert (flat[untouched] == 0).all()


@pytest.mark.parametrize("seq_lens", [[5], [1, 300, 33], [513, 1, 129, 7]])
def test_decode_attention_cpu_matches_reference(seq_lens):
    set_random_seed(0)
    num_seqs = len(seq_lens)
    num_heads = 4
    block_size = 32
    max_len = max(seq_lens)
    max_blocks = (max_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks

    kv_cache = _random_paged_cache(num_blocks * block_size, HEAD_SIZE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int64)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    scale = HEAD_SIZE**-0.5

    q = torch.randn(num_seqs, num_heads, HEAD_SIZE, dtype=DTYPE)
    kv_cache_flat = kv_cache.view(-1, 1, HEAD_SIZE)
    v_buffer = kv_cache_flat[..., :KV_LORA_RANK]

    o = torch.zeros(num_seqs, num_heads, KV_LORA_RANK, dtype=DTYPE)
    attn_logits = torch.zeros(
        num_seqs, num_heads, 4, KV_LORA_RANK + 1, dtype=torch.float32
    )
    ops.cpu_mla_decode(
        q,
        kv_cache_flat,
        v_buffer,
        o,
        None,
        None,
        None,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens_t,
        scale,
        0.0,
        False,
        0,
        None,
        None,
    )

    flat_cache = _flatten_cache(kv_cache)
    for i, seq_len in enumerate(seq_lens):
        token_ids = req_to_token[i, :seq_len]
        keys = flat_cache[token_ids]
        ref = _ref_latent_attn(q[i], keys, scale, KV_LORA_RANK)
        torch.testing.assert_close(o[i], ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "prefix_lens,extend_lens",
    [
        ([0], [9]),  # fresh prefill, no cached prefix
        ([64], [17]),  # continuation of a cached prefix
        ([0, 40, 128], [23, 5, 31]),  # mixed batch
    ],
)
def test_extend_attention_cpu_matches_reference(prefix_lens, extend_lens):
    set_random_seed(0)
    num_seqs = len(prefix_lens)
    num_heads = 4
    block_size = 32
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    max_len = max(seq_lens)
    max_blocks = (max_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks

    kv_cache = _random_paged_cache(num_blocks * block_size, HEAD_SIZE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int64)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    extend_seq_lens_t = torch.tensor(extend_lens, dtype=torch.int64)
    extend_start_loc = torch.cumsum(
        torch.tensor([0, *extend_lens[:-1]], dtype=torch.int64), dim=0
    )
    scale = HEAD_SIZE**-0.5
    total_new_tokens = sum(extend_lens)

    # The new tokens' latent K/V, already written into the cache (mirrors
    # do_kv_cache_update running before forward_mha in real usage).
    k_extend_flat = torch.randn(total_new_tokens, HEAD_SIZE, dtype=DTYPE)
    flat_cache = _flatten_cache(kv_cache)
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        new_token_ids = req_to_token[i, prefix_len : prefix_len + ext_len]
        start = extend_start_loc[i].item()
        flat_cache[new_token_ids] = k_extend_flat[start : start + ext_len]

    q_extend = torch.randn(total_new_tokens, num_heads, HEAD_SIZE, dtype=DTYPE)
    k_extend = k_extend_flat.unsqueeze(1)
    v_extend = k_extend[..., :KV_LORA_RANK]
    kv_cache_flat = kv_cache.view(-1, 1, HEAD_SIZE)
    v_buffer = kv_cache_flat[..., :KV_LORA_RANK]

    o_extend = torch.empty(total_new_tokens, num_heads, KV_LORA_RANK, dtype=DTYPE)
    ops.cpu_mla_extend(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        kv_cache_flat,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens_t,
        extend_seq_lens_t,
        extend_start_loc,
        max(extend_lens),
        scale,
        0.0,
        False,
        0,
        None,
        None,
        None,
    )

    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        start = extend_start_loc[i].item()
        for t in range(ext_len):
            visible = prefix_len + t + 1
            token_ids = req_to_token[i, :visible]
            keys = flat_cache[token_ids]
            ref = _ref_latent_attn(q_extend[start + t], keys, scale, KV_LORA_RANK)
            torch.testing.assert_close(o_extend[start + t], ref, atol=ATOL, rtol=RTOL)


class _FakeLinear:
    """Minimal duck-typed stand-in for kv_b_proj: only .weight and
    .quant_method are read by get_and_maybe_dequant_weights()."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.quant_method = None


def _make_amx_mla_impl(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    weight = torch.randn(
        num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank, dtype=DTYPE
    )
    kv_b_proj = _FakeLinear(weight)
    impl = AMXMLAImpl(
        num_heads=num_heads,
        head_size=kv_lora_rank + QK_ROPE_HEAD_DIM,
        scale=(qk_nope_head_dim + QK_ROPE_HEAD_DIM) ** -0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=qk_nope_head_dim + QK_ROPE_HEAD_DIM,
        v_head_dim=v_head_dim,
        kv_b_proj=kv_b_proj,
    )
    impl.process_weights_after_loading(DTYPE)
    return impl, weight


class _FakePrefillMetadata:
    def __init__(
        self, block_table, cpu_seq_lens, query_start_loc, req_to_token, req_pool_indices
    ):
        self.block_table = block_table
        self.cpu_seq_lens = cpu_seq_lens
        query_start_loc_i64 = query_start_loc.to(torch.int64)
        extend_seq_lens = query_start_loc_i64[1:] - query_start_loc_i64[:-1]
        self.extend_seq_lens = extend_seq_lens
        self.extend_start_loc = query_start_loc_i64[:-1]
        self.max_len_extend = int(extend_seq_lens.max().item())
        self.req_to_token = req_to_token
        self.req_pool_indices = req_pool_indices


class _FakeDecodeMetadata:
    def __init__(
        self, block_table, seq_lens, req_to_token, req_pool_indices, num_heads=4
    ):
        self.block_table = block_table
        self.seq_lens_i64 = seq_lens.to(torch.int64)
        self.req_to_token = req_to_token
        self.req_pool_indices = req_pool_indices
        self.num_kv_splits = _compute_num_kv_splits(
            block_table.size(0),
            num_heads,
            int(seq_lens.max().item()),
            current_platform.num_compute_units(),
        )


class _FakeAttnMetadata:
    def __init__(self, decode=None, prefill=None, max_seq_len=0):
        self.decode = decode
        self.prefill = prefill
        self.max_seq_len = max_seq_len


def test_amx_mla_impl_forward_mqa_matches_reference(default_vllm_config):
    """forward_mqa receives already-absorbed Q (as MLAAttention.forward_impl
    would produce via layer.W_UK_T) and must reproduce plain per-head MLA
    decode attention through W_UK/W_UV."""
    set_random_seed(1)
    num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = 4, 32, 32, 64
    impl, kv_b_weight = _make_amx_mla_impl(
        num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )
    w_uk, w_uv = kv_b_weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    ).split([qk_nope_head_dim, v_head_dim], dim=-1)

    seq_lens = [1, 40, 5]
    num_seqs = len(seq_lens)
    block_size = 32
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    head_size = kv_lora_rank + QK_ROPE_HEAD_DIM

    kv_cache = torch.randn(num_blocks, block_size, head_size, dtype=DTYPE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)

    q_nope = torch.randn(num_seqs, num_heads, qk_nope_head_dim, dtype=DTYPE)
    q_pe = torch.randn(num_seqs, num_heads, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    # Pre-absorb, mirroring MLAAttention.forward_impl's own bmm against W_UK_T.
    ql_nope = torch.einsum("bnp,lnp->bnl", q_nope.float(), w_uk.float()).to(DTYPE)

    attn_metadata = _FakeAttnMetadata(
        decode=_FakeDecodeMetadata(
            block_table=block_table,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            req_to_token=req_to_token,
            req_pool_indices=torch.arange(num_seqs, dtype=torch.int64),
        ),
        max_seq_len=max(seq_lens),
    )
    o, lse = impl.forward_mqa((ql_nope, q_pe), kv_cache, attn_metadata, layer=None)
    assert lse is None
    # De-absorb, mirroring MLAAttention.forward_impl's own bmm against
    # layer._v_up_proj, to get back into real per-head V space for
    # comparison against a from-scratch reference.
    out_real = torch.einsum("bnl,lnv->bnv", o.float(), w_uv.float())

    flat_cache = _flatten_cache(kv_cache)
    scale = impl.scale
    for i, seq_len in enumerate(seq_lens):
        token_ids = req_to_token[i, :seq_len]
        keys_latent = flat_cache[token_ids]  # (L, head_size)
        k_nope_latent, k_pe = keys_latent.split(
            [kv_lora_rank, QK_ROPE_HEAD_DIM], dim=-1
        )
        k_real = torch.einsum("lp,pnd->nld", k_nope_latent.float(), w_uk.float())
        k_pe_b = k_pe.float().unsqueeze(0).expand(num_heads, -1, -1)
        k_full = torch.cat([k_real, k_pe_b], dim=-1)  # (num_heads, L, qk_head_dim)
        q_full = torch.cat(
            [q_nope[i], q_pe[i]], dim=-1
        ).float()  # (num_heads, qk_head_dim)
        scores = torch.einsum("nd,nld->nl", q_full, k_full) * scale
        probs = torch.softmax(scores, dim=-1)
        v_real = torch.einsum("lp,pnv->nlv", k_nope_latent.float(), w_uv.float())
        ref = torch.einsum("nl,nlv->nv", probs, v_real)
        torch.testing.assert_close(out_real[i], ref, atol=_IMPL_ATOL, rtol=_IMPL_RTOL)


def test_compute_num_kv_splits():
    # A single short request still fills the threads; a full batch needs no
    # split; the split count never exceeds one per _MIN_WORK_PER_SPLIT tokens.
    assert _compute_num_kv_splits(1, 32, 2048, 64) == 16
    assert _compute_num_kv_splits(64, 32, 2048, 64) == 1
    assert _compute_num_kv_splits(1, 32, 64, 64) == 2
    assert _compute_num_kv_splits(1, 32, 1, 64) == 1


def test_amx_mla_impl_forward_decode_amx_matches_forward_mqa(default_vllm_config):
    """The fused decode path (absorb, kernel, de-absorb) must agree with the
    generic one that MLAAttention.forward_impl composes around forward_mqa."""
    set_random_seed(2)
    num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = 4, 32, 32, 64
    impl, kv_b_weight = _make_amx_mla_impl(
        num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )
    w_uk, w_uv = kv_b_weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    ).split([qk_nope_head_dim, v_head_dim], dim=-1)

    seq_lens = [1, 40, 5]
    num_seqs = len(seq_lens)
    block_size = 32
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    head_size = kv_lora_rank + QK_ROPE_HEAD_DIM
    kv_cache = torch.randn(num_blocks, block_size, head_size, dtype=DTYPE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    attn_metadata = _FakeAttnMetadata(
        decode=_FakeDecodeMetadata(
            block_table=block_table,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            req_to_token=_expand_block_table(block_table, block_size),
            req_pool_indices=torch.arange(num_seqs, dtype=torch.int64),
        ),
        max_seq_len=max(seq_lens),
    )

    q = torch.randn(
        num_seqs, num_heads, qk_nope_head_dim + QK_ROPE_HEAD_DIM, dtype=DTYPE
    )
    q_nope, q_pe = q.split([qk_nope_head_dim, QK_ROPE_HEAD_DIM], dim=-1)
    ql_nope = torch.einsum("bnp,lnp->bnl", q_nope.float(), w_uk.float()).to(DTYPE)
    o, _ = impl.forward_mqa((ql_nope, q_pe), kv_cache, attn_metadata, layer=None)
    expected = torch.einsum("bnl,lnv->bnv", o.float(), w_uv.float())

    out = torch.empty(num_seqs, num_heads * v_head_dim, dtype=DTYPE)
    impl.forward_decode_amx(q, kv_cache, attn_metadata, out)
    torch.testing.assert_close(
        out.view(num_seqs, num_heads, v_head_dim).float(),
        expected,
        atol=_IMPL_ATOL,
        rtol=_IMPL_RTOL,
    )


def _quantize_fp8_blocks(
    w: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float -> block-scaled FP8 weight and its [ceil(N/b), ceil(K/b)]
    scales, as in the DeepSeek checkpoints."""
    n, k = w.shape
    scales = torch.empty(-(-n // block), -(-k // block))
    wq = torch.empty(n, k, dtype=torch.float8_e4m3fn)
    for i in range(scales.size(0)):
        for j in range(scales.size(1)):
            rows = slice(i * block, (i + 1) * block)
            cols = slice(j * block, (j + 1) * block)
            scale = w[rows, cols].abs().max().clamp(min=1e-6) / 448
            scales[i, j] = scale
            wq[rows, cols] = (w[rows, cols] / scale).to(torch.float8_e4m3fn)
    return wq, scales


def _dequantize_fp8_blocks(
    wq: torch.Tensor, scales: torch.Tensor, block: int = 128
) -> torch.Tensor:
    n, k = wq.shape
    full = scales.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return wq.float() * full


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        DTYPE
    )


def test_mla_decode_fused_cpu_matches_reference():
    """One fused op (qkv projections, layernorms, rope, W_UK absorption, KV
    write, decode attention, W_UV de-absorption) against the same pipeline
    written with plain torch ops on dequantized weights."""
    set_random_seed(3)
    hidden, q_lora, kv_lora, rope = 256, 128, 128, 64
    num_heads, nope, v_head_dim = 4, 64, 64
    qk_head_dim = nope + rope
    kv_dim = kv_lora + rope
    eps = 1e-6
    block_size = 32
    seq_lens = [1, 40, 77]
    num_seqs = len(seq_lens)

    x = torch.randn(num_seqs, hidden, dtype=DTYPE)
    w_q_a, s_q_a = _quantize_fp8_blocks(torch.randn(q_lora, hidden) * 0.1)
    w_kv_a, s_kv_a = _quantize_fp8_blocks(torch.randn(kv_dim, hidden) * 0.1)
    w_q_b, s_q_b = _quantize_fp8_blocks(
        torch.randn(num_heads * qk_head_dim, q_lora) * 0.1
    )
    qkv_a_packed = torch.ops._C.convert_weight_packed(torch.cat([w_q_a, w_kv_a]))
    qkv_a_scale = torch.cat([s_q_a, s_kv_a])
    q_b_packed = torch.ops._C.convert_weight_packed(w_q_b)
    w_kc = torch.randn(num_heads, kv_lora, nope, dtype=DTYPE) * 0.1
    w_vc = torch.randn(num_heads, v_head_dim, kv_lora, dtype=DTYPE) * 0.1
    q_ln = torch.rand(q_lora, dtype=DTYPE) + 0.5
    kv_ln = torch.rand(kv_lora, dtype=DTYPE) + 0.5
    positions = torch.tensor([n - 1 for n in seq_lens], dtype=torch.int64)
    cos_sin = (torch.rand(max(seq_lens), rope) * 2 - 1).to(DTYPE)

    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    slot_mapping = req_to_token[torch.arange(num_seqs), positions]
    kv_cache = torch.randn(num_blocks, block_size, kv_dim, dtype=DTYPE)
    ref_cache = _flatten_cache(kv_cache.clone())
    sm_scale = qk_head_dim**-0.5

    # reference
    qa = (
        x.float() @ _dequantize_fp8_blocks(torch.cat([w_q_a, w_kv_a]), qkv_a_scale).T
    ).to(DTYPE)
    q_c = _rms_norm(qa[:, :q_lora], q_ln, eps)
    kv_c = _rms_norm(qa[:, q_lora : q_lora + kv_lora], kv_ln, eps)
    k_pe = qa[:, q_lora + kv_lora :]
    q = (q_c.float() @ _dequantize_fp8_blocks(w_q_b, s_q_b).T).to(DTYPE)
    q = q.view(num_seqs, num_heads, qk_head_dim)

    def apply_rope(v: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cos, sin = cos_sin[pos].float().chunk(2, dim=-1)
        v = v.float()
        while cos.dim() < v.dim():
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        out = torch.empty_like(v)
        out[..., 0::2] = v[..., 0::2] * cos - v[..., 1::2] * sin
        out[..., 1::2] = v[..., 1::2] * cos + v[..., 0::2] * sin
        return out.to(DTYPE)

    q_pe = apply_rope(q[..., nope:], positions)
    k_pe = apply_rope(k_pe, positions)
    ql = torch.einsum("bhp,hlp->bhl", q[..., :nope].float(), w_kc.float()).to(DTYPE)
    ref_cache[slot_mapping] = torch.cat([kv_c, k_pe], dim=-1)
    expected = torch.empty(num_seqs, num_heads, v_head_dim)
    for i, seq_len in enumerate(seq_lens):
        keys = ref_cache[req_to_token[i, :seq_len]].float()
        q_full = torch.cat([ql[i], q_pe[i]], dim=-1).float()
        probs = torch.softmax(q_full @ keys.T * sm_scale, dim=-1)
        o = probs @ keys[:, :kv_lora]
        expected[i] = torch.einsum("hl,hvl->hv", o, w_vc.float())

    out = torch.empty(num_seqs, num_heads * v_head_dim, dtype=DTYPE)
    ops.mla_decode_fused_cpu(
        x,
        qkv_a_packed,
        q_b_packed,
        torch.ops._C.convert_weight_packed(w_kc),
        torch.ops._C.convert_weight_packed(w_vc),
        q_ln,
        kv_ln,
        positions,
        cos_sin,
        eps,
        qkv_a_scale,
        s_q_b,
        None,
        [128, 128],
        q_lora,
        kv_lora,
        rope,
        kv_cache,
        slot_mapping,
        req_to_token,
        torch.arange(num_seqs, dtype=torch.int64),
        torch.tensor(seq_lens, dtype=torch.int64),
        _compute_num_kv_splits(
            num_seqs,
            num_heads,
            max(seq_lens),
            current_platform.num_compute_units(),
        ),
        sm_scale,
        out,
    )
    torch.testing.assert_close(
        _flatten_cache(kv_cache)[slot_mapping],
        ref_cache[slot_mapping],
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        out.view(num_seqs, num_heads, v_head_dim).float(),
        expected,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "decode_lens, prefills, w8a8",
    [
        pytest.param([1, 40, 77], [], False, id="decode-only"),
        pytest.param([33], [], False, id="decode-single"),
        pytest.param([], [(0, 50), (20, 30)], True, id="prefill-only"),
        pytest.param([40, 77, 100], [(0, 64), (16, 24)], True, id="mixed"),
    ],
)
def test_mla_layer_cpu_matches_reference(decode_lens, prefills, w8a8):
    """The whole-layer op on decode tokens, prefill tokens (fresh and continuing
    a cached prefix) and both together, against plain torch ops on dequantized
    weights. Per-request metadata sits in fixed-capacity buffers, as in the
    backend's direct mode. Batches of more than four tokens run the W8A8 GEMMs,
    whose per-token FP8 activation quantization (3 mantissa bits) adds noise the
    reference does not model; those cases bound the RMS error relative to the
    output RMS instead of every element (measured: about 3% and 5%)."""
    set_random_seed(5)
    hidden, q_lora, kv_lora, rope = 256, 128, 128, 64
    num_heads, nope, v_head_dim = 4, 64, 64
    qk_head_dim = nope + rope
    kv_dim = kv_lora + rope
    eps = 1e-6
    block_size = 32
    sm_scale = qk_head_dim**-0.5

    # (tokens already in the cache, new tokens) per request, decode requests first
    reqs = [(n - 1, 1) for n in decode_lens] + list(prefills)
    num_decode = len(decode_lens)
    num_prefill_reqs = len(prefills)
    num_tokens = sum(q_len for _, q_len in reqs)
    num_prefill_tokens = num_tokens - num_decode
    max_total = max(ctx + q_len for ctx, q_len in reqs)
    max_blocks = -(-max_total // block_size)
    capacity = len(reqs) + 3
    max_context = max_blocks * block_size

    w_q_a, s_q_a = _quantize_fp8_blocks(torch.randn(q_lora, hidden) * 0.1)
    w_kv_a, s_kv_a = _quantize_fp8_blocks(torch.randn(kv_dim, hidden) * 0.1)
    w_q_b, s_q_b = _quantize_fp8_blocks(
        torch.randn(num_heads * qk_head_dim, q_lora) * 0.1
    )
    w_qkv_a = torch.cat([w_q_a, w_kv_a])
    s_qkv_a = torch.cat([s_q_a, s_kv_a])
    w_kc = torch.randn(num_heads, kv_lora, nope, dtype=DTYPE) * 0.1
    w_vc = torch.randn(num_heads, v_head_dim, kv_lora, dtype=DTYPE) * 0.1
    q_ln = torch.rand(q_lora, dtype=DTYPE) + 0.5
    kv_ln = torch.rand(kv_lora, dtype=DTYPE) + 0.5
    cos_sin = (torch.rand(max_total, rope) * 2 - 1).to(DTYPE)

    def prepack_w8a8(w: torch.Tensor, scales: torch.Tensor):
        scale_rows = scales.repeat_interleave(128, 0)[: w.size(0)].contiguous()
        return torch.ops._C.float8_linear_prepack_cpu(w, scale_rows)

    qkv_a_w8a8, qkv_a_s8a8 = prepack_w8a8(w_qkv_a, s_qkv_a)
    q_b_w8a8, q_b_s8a8 = prepack_w8a8(w_q_b, s_q_b)

    x = torch.randn(num_tokens, hidden, dtype=DTYPE)
    positions = torch.cat([torch.arange(ctx, ctx + q_len) for ctx, q_len in reqs]).to(
        torch.int64
    )

    num_blocks = len(reqs) * max_blocks
    block_table = torch.randperm(num_blocks).view(len(reqs), max_blocks)
    req_to_token = _expand_block_table(block_table, block_size)
    kv_cache = torch.randn(num_blocks, block_size, kv_dim, dtype=DTYPE)
    ref_cache = _flatten_cache(kv_cache.clone())
    token_req = torch.cat(
        [torch.full((q_len,), r) for r, (_, q_len) in enumerate(reqs)]
    )
    slot_mapping = req_to_token[token_req, positions]

    # reference
    qa = (x.float() @ _dequantize_fp8_blocks(w_qkv_a, s_qkv_a).T).to(DTYPE)
    q_c = _rms_norm(qa[:, :q_lora], q_ln, eps)
    kv_c = _rms_norm(qa[:, q_lora : q_lora + kv_lora], kv_ln, eps)
    q = (q_c.float() @ _dequantize_fp8_blocks(w_q_b, s_q_b).T).to(DTYPE)
    q = q.view(num_tokens, num_heads, qk_head_dim)

    def apply_rope(v: torch.Tensor) -> torch.Tensor:
        cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
        v = v.float()
        while cos.dim() < v.dim():
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        out = torch.empty_like(v)
        out[..., 0::2] = v[..., 0::2] * cos - v[..., 1::2] * sin
        out[..., 1::2] = v[..., 1::2] * cos + v[..., 0::2] * sin
        return out.to(DTYPE)

    q_pe = apply_rope(q[..., nope:])
    k_pe = apply_rope(qa[:, q_lora + kv_lora :])
    ql = torch.einsum("bhp,hlp->bhl", q[..., :nope].float(), w_kc.float()).to(DTYPE)
    ref_cache[slot_mapping] = torch.cat([kv_c, k_pe], dim=-1)
    expected = torch.empty(num_tokens, num_heads, v_head_dim)
    for t in range(num_tokens):
        keys = ref_cache[req_to_token[token_req[t], : positions[t] + 1]].float()
        q_full = torch.cat([ql[t], q_pe[t]], dim=-1).float()
        probs = torch.softmax(q_full @ keys.T * sm_scale, dim=-1)
        expected[t] = torch.einsum(
            "hl,hvl->hv", probs @ keys[:, :kv_lora], w_vc.float()
        )

    # fixed-capacity metadata buffers, first rows valid
    def padded(values, dtype, fill=0):
        buf = torch.full((capacity,), fill, dtype=dtype)
        buf[: len(values)] = torch.tensor(values, dtype=dtype)
        return buf

    rt_buf = torch.zeros(capacity, max_context + block_size, dtype=torch.int32)
    rt_buf[: len(reqs), : req_to_token.size(1)] = req_to_token.to(torch.int32)
    extend_lens = [q_len for _, q_len in prefills]
    scalars = torch.tensor(
        [
            num_decode,
            num_prefill_tokens,
            num_prefill_reqs,
            _compute_num_kv_splits(
                max(num_decode, 1),
                num_heads,
                max(decode_lens, default=1),
                current_platform.num_compute_units(),
            ),
        ],
        dtype=torch.int64,
    )
    slots = torch.full((num_tokens + 5,), -1, dtype=torch.int64)
    slots[:num_tokens] = slot_mapping

    out = torch.empty(num_tokens, num_heads * v_head_dim, dtype=DTYPE)
    ops.mla_layer_cpu(
        x,
        torch.cat([positions, torch.zeros(5, dtype=torch.int64)]),
        out,
        kv_cache,
        torch.ops._C.convert_weight_packed(w_qkv_a),
        s_qkv_a,
        torch.ops._C.convert_weight_packed(w_q_b),
        s_q_b,
        qkv_a_w8a8,
        qkv_a_s8a8,
        q_b_w8a8,
        q_b_s8a8,
        torch.ops._C.convert_weight_packed(w_kc),
        torch.ops._C.convert_weight_packed(w_vc),
        q_ln,
        kv_ln,
        cos_sin,
        None,
        scalars,
        slots,
        rt_buf,
        torch.arange(capacity, dtype=torch.int64),
        padded([n for n in decode_lens], torch.int64),
        rt_buf,
        padded(
            [num_decode + r for r in range(num_prefill_reqs)],
            torch.int64,
        ),
        padded([ctx + q_len for ctx, q_len in prefills], torch.int64),
        padded(extend_lens, torch.int32),
        padded(
            [sum(extend_lens[:r]) for r in range(num_prefill_reqs)],
            torch.int32,
        ),
        eps,
        q_lora,
        kv_lora,
        rope,
        sm_scale,
        4,
    )
    actual_rows = _flatten_cache(kv_cache)[slot_mapping]
    actual_out = out.view(num_tokens, num_heads, v_head_dim).float()
    for actual, want in (
        (actual_rows.float(), ref_cache[slot_mapping].float()),
        (actual_out, expected),
    ):
        if w8a8:
            rel_rms = (actual - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()
            assert rel_rms < 0.08, rel_rms
        else:
            torch.testing.assert_close(actual, want, atol=5e-2, rtol=5e-2)


def test_amx_mla_impl_forward_mha_matches_reference(default_vllm_config):
    """forward_mha receives raw unabsorbed Q and must attend correctly for a
    prefill batch with a mix of fresh and continued (cached-prefix)
    sequences."""
    set_random_seed(2)
    num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = 4, 32, 32, 64
    impl, kv_b_weight = _make_amx_mla_impl(
        num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )
    w_uk, w_uv = kv_b_weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    ).split([qk_nope_head_dim, v_head_dim], dim=-1)

    prefix_lens = [0, 20]
    extend_lens = [13, 7]
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    num_seqs = len(prefix_lens)
    block_size = 32
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    head_size = kv_lora_rank + QK_ROPE_HEAD_DIM

    kv_cache = torch.randn(num_blocks, block_size, head_size, dtype=DTYPE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    flat_cache = _flatten_cache(kv_cache)

    total_new_tokens = sum(extend_lens)
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(extend_lens), dim=0).tolist()],
        dtype=torch.int64,
    )

    q_nope = torch.randn(total_new_tokens, num_heads, qk_nope_head_dim, dtype=DTYPE)
    q_pe = torch.randn(total_new_tokens, num_heads, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    q = torch.cat([q_nope, q_pe], dim=-1)

    kv_c_normed = torch.randn(total_new_tokens, kv_lora_rank, dtype=DTYPE)
    k_pe = torch.randn(total_new_tokens, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE)

    # Write the new tokens' latent K/V into the cache ahead of time, mirroring
    # do_kv_cache_update running before forward_mha in real usage.
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        new_token_ids = req_to_token[i, prefix_len : prefix_len + ext_len]
        start = query_start_loc[i].item()
        flat_cache[new_token_ids, :kv_lora_rank] = kv_c_normed[start : start + ext_len]
        flat_cache[new_token_ids, kv_lora_rank:] = k_pe[start : start + ext_len, 0]

    attn_metadata = _FakeAttnMetadata(
        prefill=_FakePrefillMetadata(
            block_table=block_table,
            cpu_seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            query_start_loc=query_start_loc,
            req_to_token=req_to_token,
            req_pool_indices=torch.arange(num_seqs, dtype=torch.int64),
        )
    )
    output = torch.empty(total_new_tokens, num_heads * v_head_dim, dtype=DTYPE)
    impl.forward_mha(
        q, kv_c_normed, k_pe, kv_cache, attn_metadata, k_scale=None, output=output
    )
    out_view = output.view(total_new_tokens, num_heads, v_head_dim).float()

    scale = impl.scale
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        start = query_start_loc[i].item()
        for t in range(ext_len):
            visible = prefix_len + t + 1
            token_ids = req_to_token[i, :visible]
            keys_latent = flat_cache[token_ids]
            k_nope_latent, k_pe_ref = keys_latent.split(
                [kv_lora_rank, QK_ROPE_HEAD_DIM], dim=-1
            )
            k_real = torch.einsum("lp,pnd->nld", k_nope_latent.float(), w_uk.float())
            k_pe_b = k_pe_ref.float().unsqueeze(0).expand(num_heads, -1, -1)
            k_full = torch.cat([k_real, k_pe_b], dim=-1)
            q_full = torch.cat([q_nope[start + t], q_pe[start + t]], dim=-1).float()
            scores = torch.einsum("nd,nld->nl", q_full, k_full) * scale
            probs = torch.softmax(scores, dim=-1)
            v_real = torch.einsum("lp,pnv->nlv", k_nope_latent.float(), w_uv.float())
            ref = torch.einsum("nl,nlv->nv", probs, v_real)
            torch.testing.assert_close(
                out_view[start + t], ref, atol=_IMPL_ATOL, rtol=_IMPL_RTOL
            )
