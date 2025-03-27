# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from typing import Any, Dict

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.chunked_prefill = True
        super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert not self.use_spec_decode, "spec decode is not supported."
        assert not self.model_config.uses_mrope, "mrope is not supported."
        assert self.lora_config is None, "lora is not supported."

        self.use_cuda_graph = False

        self._postprocess_tenosrs()

        self.seq_start_loc_cpu = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def _postprocess_tenosrs(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str,
                           device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                print(cpu_attr_name, device_attr_name)
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for k, v in vars(self).items():
            if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                replace_tensor(self, k, k[:-4])

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for k, v in vars(self.input_batch.block_table).items():
            if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch.block_table, k, k[:-4])

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings():
            self._dummy_run(self.max_num_tokens)
        logger.info("Warming up done.")

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: Dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                        dtype=dtype,
                                                        device=self.device)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)


@contextmanager
def _set_global_compilation_settings():
    import torch._inductor.config

    # Note: The CPPGEMM backend requires freezing parameters.
    freezing_value = torch._inductor.config.freezing
    torch._inductor.config.freezing = True
    # Note: workaround for "ValueError: fast mode: can't pickle cyclic objects including object type dict"
    force_disable_caches = torch._inductor.config.force_disable_caches
    torch._inductor.config.force_disable_caches = True
    yield
    torch._inductor.config.freezing = freezing_value
    torch._inductor.config.force_disable_caches = force_disable_caches
