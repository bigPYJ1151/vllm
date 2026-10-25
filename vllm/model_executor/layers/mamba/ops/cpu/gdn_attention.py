# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_CPU_GDN_ATTENTION_OPS_REGISTERED = False


def is_arm_bf16() -> bool:
    return (
        current_platform.get_cpu_architecture() == CpuArchEnum.ARM
        and torch.cpu.get_capabilities().get("bf16", False)
    )


def _check_supported(layer, mixed_qkv: torch.Tensor) -> None:
    """Fail loudly for setups the fused C++ GDN op does not cover."""
    assert mixed_qkv.dtype == torch.bfloat16, (
        f"CPU GDN attention requires BF16 (got {mixed_qkv.dtype})."
    )
    assert torch.cpu._is_avx512_bf16_supported() or is_arm_bf16(), (
        "CPU GDN attention requires AVX512-BF16 or ARM BF16."
    )
    assert not is_conv_state_dim_first(), (
        "CPU GDN attention requires the SD conv state layout."
    )
    width = layer.conv1d.weight.size(-1)
    assert layer.kv_cache[0].size(-2) == width - 1, (
        "CPU GDN attention does not support speculative decoding."
    )
    layer._cpu_gdn_checked = True


def cpu_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """CPU custom op for the core GDN attention computation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]

    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return

    assert isinstance(attn_metadata, dict)
    attn_metadata_i = attn_metadata[layer.prefix]
    assert isinstance(attn_metadata_i, GDNAttentionMetadata)

    if attn_metadata_i.num_actual_tokens == 0:
        return

    if not getattr(layer, "_cpu_gdn_checked", False):
        _check_supported(layer, mixed_qkv)

    assert (
        attn_metadata_i.spec_sequence_masks is None
        and attn_metadata_i.num_accepted_tokens is None
    ), "speculative decode is not supported by CPU GDN attention."

    state_indices = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices is not None
    assert query_start_loc is not None

    torch.ops._C.gdn_attention_core_cpu(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        layer.kv_cache[0],
        layer.conv1d.weight,
        layer.conv1d.bias,
        layer.kv_cache[1],
        layer.A_log,
        layer.dt_bias,
        state_indices,
        query_start_loc,
        attn_metadata_i.has_initial_state,
        attn_metadata_i.num_decodes,
        attn_metadata_i.num_decode_tokens,
        attn_metadata_i.num_prefills,
        attn_metadata_i.num_prefill_tokens,
        layer.head_k_dim,
        layer.head_v_dim,
        layer.activation == "silu",
    )


def register_cpu_gdn_attention_ops() -> None:
    global _CPU_GDN_ATTENTION_OPS_REGISTERED
    if _CPU_GDN_ATTENTION_OPS_REGISTERED:
        return

    direct_register_custom_op(
        op_name="cpu_gdn_attention_core",
        op_func=cpu_gdn_attention_core,
        mutates_args=["core_attn_out"],
    )
    _CPU_GDN_ATTENTION_OPS_REGISTERED = True
