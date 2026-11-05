# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.cpu_attn import init_direct_forward_layers
from vllm.v1.attention.backends.mla.amx_mla import init_direct_mla_layers
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        if envs.VLLM_CPU_ATTN_DIRECT:
            register_backend(
                MambaAttentionBackendEnum.GDN_ATTN,
                "vllm.v1.attention.backends.cpu_gdn_attn.CPUGDNAttentionBackend",
                is_mamba=True,
            )
        super().__init__(*args, **kwargs)

    def add_requests(self, scheduler_output) -> None:
        # Without new requests the base method only re-applies unchanged
        # sampler state.
        if scheduler_output.scheduled_new_reqs:
            super().add_requests(scheduler_output)

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Layers called directly from compiled code need their metadata
        # attribute to exist when the model is traced.
        layers = self.vllm_config.compilation_config.static_forward_context.values()
        init_direct_forward_layers(layers)
        init_direct_mla_layers(layers, self.vllm_config)
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
