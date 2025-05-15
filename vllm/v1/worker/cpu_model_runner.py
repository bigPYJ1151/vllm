from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch

from vllm.attention.backends.torch_sdpa import (TorchSDPABackend,
                                                TorchSDPAMetadata)
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import bind_kv_cache
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.cpu_input_batch import CPUInputBatch
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

        self.input_batch: CPUInputBatch = CPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )
        self.use_cuda_graph = False

        self.input_ids = self.input_ids_cpu
        self.positions = self.positions_cpu

        self.seq_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                        dtype=torch.int32,
                                        device="cpu",)
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()


    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_states(scheduler_output)
        self.input_batch.reorder()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings():
            self._dummy_run(max(16, self.max_num_reqs))
        logger.info("Warming up done.")

    def _prepare_inputs(self, scheduler_output: SchedulerOutput):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        seq_lens_np = self.seq_lens_np[:num_reqs]
        np.add(self.input_batch.num_computed_tokens_cpu[:num_reqs],
               num_scheduled_tokens,
               out=seq_lens_np)
        max_seq_len = seq_lens_np.max().item()
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=self.seq_start_loc_np[1:num_reqs + 1])

        num_prompt_reqs = self.input_batch.num_prompt_req
        num_prefill_tokens = self.query_start_loc_np[num_prompt_reqs].item()
        num_decode_tokens = self.query_start_loc_np[num_reqs].item(
        ) - num_prefill_tokens
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].long(
        )
        max_query_len = num_scheduled_tokens.max().item()  # type: ignore

        attn_metadata = TorchSDPAMetadata(
            num_prefills=num_prompt_reqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=self.
            seq_lens_cpu[num_prompt_reqs:num_reqs],  # decode
            max_decode_seq_len=max_seq_len,  # decode
            block_tables=self.input_batch.
            block_table.get_device_tensor()[num_prompt_reqs:num_reqs],  # decode
            chunked_prefill=True,
            max_query_len=max_query_len,
            max_kv_len=max_seq_len,
            query_start_loc=self.query_start_loc_cpu[:num_prompt_reqs +
                                                     1],  # prefill
            kv_start_loc=self.seq_start_loc_cpu[:num_prompt_reqs +
                                                1],  # prefill
            prefill_block_tables=self.input_batch.
            block_table.get_device_tensor()[:num_prompt_reqs],  # prefill
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )

        query_start_loc = self.query_start_loc_cpu[:num_reqs + 1]

        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return attn_metadata, logits_indices, None

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