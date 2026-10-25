# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN attention backend for CPU layers that are called directly from
compiled code (``VLLM_CPU_ATTN_DIRECT``).

The layers read their per-step metadata from ``layer.cpu_gdn_metadata`` instead
of the forward context, so the builder refreshes one shared instance per group
at every build.
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass
class CPUGDNDirectMetadata:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    has_initial_state: torch.Tensor
    # [num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens]
    counts: torch.Tensor


def placeholder_gdn_metadata() -> CPUGDNDirectMetadata:
    """Metadata of a step without requests; the direct forward skips it."""
    return CPUGDNDirectMetadata(
        state_indices=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        has_initial_state=torch.zeros(0, dtype=torch.bool),
        counts=torch.zeros(4, dtype=torch.int32),
    )


class CPUGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert self.num_spec == 0, "speculative decoding is not supported"
        # One instance shared by all layers of the group, refreshed in place.
        self._direct_metadata = placeholder_gdn_metadata()
        for name in layer_names:
            layer = vllm_config.compilation_config.static_forward_context[name]
            assert isinstance(layer, GatedDeltaNetAttention)
            layer.cpu_gdn_metadata = self._direct_metadata

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        assert metadata.spec_sequence_masks is None
        direct = self._direct_metadata
        direct.state_indices = metadata.non_spec_state_indices_tensor
        direct.query_start_loc = metadata.non_spec_query_start_loc
        direct.has_initial_state = (
            metadata.has_initial_state
            if metadata.has_initial_state is not None
            else direct.has_initial_state[:0]
        )
        direct.counts = torch.tensor(
            [
                metadata.num_decodes,
                metadata.num_decode_tokens,
                metadata.num_prefills,
                metadata.num_prefill_tokens,
            ],
            dtype=torch.int32,
        )
        return metadata


class CPUGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[CPUGDNAttentionMetadataBuilder]:
        return CPUGDNAttentionMetadataBuilder
