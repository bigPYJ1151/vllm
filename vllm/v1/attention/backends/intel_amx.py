# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm import _custom_ops as ops


class IntelAMXBackend(AttentionBackend):
    accept_output_buffer: bool = True 

    @staticmethod
    def get_name() -> str:
        return "Intel_AMX_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["IntelAMXImpl"]:
        return IntelAMXImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return IntelAMXMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["IntelAMXMetadataBuilder"]:
        return IntelAMXMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks * block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not supported.")

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

@dataclass
class IntelAMXMetadata:
    # KV cache slot of each tokens
    slot_mapping: torch.Tensor
    prefill_tokens: int
    prefill_reqs: int
    decode_tokens: int 
    # [max_req_num, max_context_len], unrolled block table
    req_to_token: torch.Tensor 
    # [req_num], index for req_to_token
    req_pool_indices: torch.Tensor
    # [req_num], token nums for each req
    seq_lens: torch.Tensor
    # [req_num], token nums for each reqs in prefill batch
    extend_seq_lens: torch.Tensor
    # [req_num], token start offset for each reqs in prefill batch
    query_start_loc: torch.Tensor
    max_len_extend: int
    # [num_seqs, num_heads, num_kv_splits, head_size_v + 1], logits cache
    attn_logits: torch.Tensor
    
class IntelAMXMetadataBuilder:

    def __init__(self, runner: CPUModelRunner) -> None:
        self.runner = runner

        # For reorder
        self.reorder_prompt_req_index_list = np.empty(self.runner.max_num_reqs,
                                                      dtype=np.int64)
        self.reorder_decode_req_index_list = np.empty(self.runner.max_num_reqs,
                                                      dtype=np.int64)
        self.num_prompt_req: int = 0
        self.req_to_token_tensor = torch.empty((runner.max_num_reqs, runner.max_model_len), dtype=torch.int32)
        self.req_to_token_np = self.req_to_token_tensor.numpy()
        self.block_size = runner.block_size
        assert runner.max_model_len % runner.block_size == 0, f"max_model_len({runner.max_model_len}) should be multple of block size({runner.block_size})"
        self.block_offsets_np = np.arange(0, self.block_size, dtype=np.int32)[None, None, :]

        self.req_pool_indice_tensor = torch.arange(0, runner.max_num_reqs, dtype=torch.int64)
        self.attn_logits_tensor = torch.empty((runner.max_num_reqs, runner.num_query_heads, 8, runner.vllm_config.model_config.get_head_size() + 1), dtype=torch.float32)

        self.query_lens_tensor = torch.empty(runner.max_num_reqs, dtype=torch.int32)
        self.query_lens_np = self.query_lens_tensor.numpy()

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        prompt_list_idx = 0
        decode_list_idx = 0
        for req_index in range(input_batch.num_reqs):
            if input_batch.num_computed_tokens_cpu[
                    req_index] < input_batch.num_prompt_tokens[req_index]:
                # prompt stage
                self.reorder_prompt_req_index_list[prompt_list_idx] = req_index
                prompt_list_idx += 1
            else:
                # decode stage
                self.reorder_decode_req_index_list[decode_list_idx] = req_index
                decode_list_idx += 1
        assert decode_list_idx + prompt_list_idx == input_batch.num_reqs

        # Update prompt requests number
        self.num_prompt_req = prompt_list_idx

        reorder_req_num = 0
        for req_index in range(decode_list_idx):
            if self.reorder_decode_req_index_list[req_index] < prompt_list_idx:
                reorder_req_num += 1
            else:
                break

        if reorder_req_num == 0:
            return False

        reorder_prompt_list = (
            self.reorder_prompt_req_index_list[:prompt_list_idx]
            [-reorder_req_num:])
        reorder_decode_list = (
            self.reorder_decode_req_index_list[:decode_list_idx]
            [:reorder_req_num])
        assert reorder_decode_list.size == reorder_prompt_list.size

        for idx in range(reorder_req_num):
            prompt_req_index = reorder_prompt_list[idx].item()
            decode_req_index = reorder_decode_list[idx].item()
            input_batch.swap_states(prompt_req_index, decode_req_index)

        return True

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        runner = self.runner
        seq_lens_np = runner.seq_lens_np[:num_reqs]
        num_prompt_req = self.num_prompt_req

        slot_mapping = runner.slot_mapping_cpu[:num_actual_tokens]
        num_prefill_tokens = runner.query_start_loc_np[num_prompt_req].item()
        num_decode_tokens = runner.query_start_loc_np[num_reqs].item(
        ) - num_prefill_tokens
        max_seq_block_num = cdiv(seq_lens_np.max().item(), self.block_size)
        max_ceil_seq_len = max_seq_block_num * self.block_size
        np.add(
            runner.input_batch.block_table.block_table_np[:num_reqs, :max_seq_block_num, None] * self.block_size,
            self.block_offsets_np,
            out=self.req_to_token_np[:num_reqs, :max_ceil_seq_len].reshape((num_reqs, max_seq_block_num, self.block_size)),
        )
        np.subtract(runner.query_start_loc_np[1:num_reqs+1], runner.query_start_loc_np[:num_reqs], out=self.query_lens_np[:num_reqs])

        attn_metadata = IntelAMXMetadata(
            slot_mapping=slot_mapping,
            prefill_tokens=num_prefill_tokens,
            prefill_reqs=num_prompt_req,
            decode_tokens=num_decode_tokens,
            req_to_token=self.req_to_token_tensor,
            req_pool_indices=self.req_pool_indice_tensor[:num_reqs],
            seq_lens=runner.seq_lens_cpu[:num_reqs].long(),
            extend_seq_lens=self.query_lens_tensor[:num_prompt_req],
            query_start_loc=runner.query_start_loc_cpu[:num_reqs+1],
            max_len_extend=max_query_len,
            attn_logits=self.attn_logits_tensor[num_prompt_req:num_reqs],
        )

        return attn_metadata

class IntelAMXImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "IntelAMX does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            raise ValueError(
                "IntelAMX does not support alibi_slopes.")
        if sliding_window is not None:
            raise ValueError(
                "IntelAMX does not support sliding_window.") 

        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IntelAMX")
        if use_irope:
            raise ValueError(
                "IntelAMX does not support local attention.")

        if is_quantized_kv_cache(self.kv_cache_dtype): 
            raise NotImplementedError(
                "IntelAMX does not support fp8 kv-cache on this device.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IntelAMXMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IntelAMX backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks * block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # For warming up.
            return output

        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C.set_kv_buffer(
            key_cache,
            value_cache,
            key,
            value,
            attn_metadata.slot_mapping,
        )

        prefill_reqs = attn_metadata.prefill_reqs
        prefill_tokens = attn_metadata.prefill_tokens
        decode_tokens = attn_metadata.decode_tokens
        if prefill_tokens > 0:
            torch.ops._C.extend_attention_cpu(
                query[:prefill_tokens].contiguous(),
                key[:prefill_tokens],
                value[:prefill_tokens],
                output[:prefill_tokens],
                key_cache,
                value_cache,
                attn_metadata.req_to_token,
                attn_metadata.req_pool_indices[:prefill_reqs],
                attn_metadata.seq_lens[:prefill_reqs],
                attn_metadata.extend_seq_lens[:prefill_reqs],
                attn_metadata.query_start_loc[:prefill_reqs],
                attn_metadata.max_len_extend,
                self.scale,
                self.logits_soft_cap,
            )

        if decode_tokens > 0:
            torch.ops._C.decode_attention_cpu(
                query[prefill_tokens:].contiguous(),
                key_cache,
                value_cache,
                output[prefill_tokens:],
                key[prefill_tokens:],
                value[prefill_tokens:],
                attn_metadata.slot_mapping[prefill_tokens:],
                attn_metadata.attn_logits,
                attn_metadata.req_to_token,
                attn_metadata.req_pool_indices[prefill_reqs:],
                attn_metadata.seq_lens[prefill_reqs:],
                self.scale,
                self.logits_soft_cap,
            )

        return output
