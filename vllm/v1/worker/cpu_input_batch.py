from typing import Dict, List, Optional, cast

import numpy as np
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch


class CPUInputBatch(InputBatch):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        assert device == torch.device("cpu")
        assert not pin_memory
        super().__init__(max_num_reqs, max_model_len, max_num_blocks_per_req,
                         device, pin_memory, vocab_size)

        # reset tensors
        self.block_table.block_table = self.block_table.block_table_cpu

        self.temperature = self.temperature_cpu_tensor
        self.top_p = self.top_p_cpu_tensor
        self.top_k = self.top_k_cpu_tensor
        self.min_p = self.min_p_cpu_tensor
        self.frequency_penalties = self.frequency_penalties_cpu_tensor
        self.presence_penalties = self.presence_penalties_cpu_tensor
        self.repetition_penalties = self.repetition_penalties_cpu_tensor

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = self.allowed_token_ids_mask_cpu_tensor

        # For reorder
        self.reorder_prompt_req_index_list = np.empty(self.max_num_reqs,
                                                      dtype=np.int64)
        self.reorder_decode_req_index_list = np.empty(self.max_num_reqs,
                                                      dtype=np.int64)
        self.num_prompt_req: int = 0

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if self.all_greedy:
            temperature = None
        else:
            temperature = self.temperature_cpu_tensor[:num_reqs]

        if not self.no_penalties:
            # The prompt tokens are used only for applying penalties during
            # the sampling process. Hence copy these tensors only when
            # there are requests which need penalties to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: Optional[torch.Tensor] = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            min_p=None if self.no_min_p else self.min_p[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            min_tokens=self.min_tokens,
            no_penalties=self.no_penalties,
            logit_bias=self.logit_bias[:num_reqs],
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
        )

    def _move_req(self, dst: int, src: int):
        req_id = self._req_ids[src]
        assert req_id is not None
        self._req_ids[dst] = req_id
        self._req_ids[src] = None
        self.req_id_to_index[req_id] = dst

        self.token_ids_cpu[dst] = self.token_ids_cpu[src]
        self.num_tokens[dst] = self.num_tokens[src]
        self.num_tokens_no_spec[dst] = self.num_tokens_no_spec[src]
        self.num_prompt_tokens[dst] = self.num_prompt_tokens[src]
        self.num_computed_tokens_cpu[dst] = self.num_computed_tokens_cpu[src]
        self.block_table.move_row(src, dst)
        self.temperature_cpu[dst] = self.temperature_cpu[src]
        self.top_p_cpu[dst] = self.top_p_cpu[src]
        self.top_k_cpu[dst] = self.top_k_cpu[src]
        self.min_p_cpu[dst] = self.min_p_cpu[src]
        self.frequency_penalties_cpu[dst] = self.frequency_penalties_cpu[src]
        self.presence_penalties_cpu[dst] = self.presence_penalties_cpu[src]
        self.repetition_penalties_cpu[dst] = self.repetition_penalties_cpu[src]

        min_tokens = self.min_tokens.pop(src, None)
        if min_tokens is not None:
            self.min_tokens[dst] = self.min_tokens[src]

        self.request_lora_mapping[dst] = self.request_lora_mapping[src]

        generator = self.generators.pop(src, None)
        if generator is not None:
            self.generators[dst] = generator
        self.logit_bias[dst] = self.logit_bias[src]

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[dst] = self.allowed_token_ids_mask_cpu_tensor[src]

        bad_words_token_ids = self.bad_words_token_ids.pop(src, None)
        if bad_words_token_ids is not None:
            self.bad_words_token_ids[dst] = bad_words_token_ids

    def reorder(self) -> None:
        prompt_list_idx = 0
        decode_list_idx = 0
        for req_index in range(self.num_reqs):
            if self.num_computed_tokens_cpu[
                    req_index] < self.num_prompt_tokens[req_index]:
                # prompt stage
                self.reorder_prompt_req_index_list[prompt_list_idx] = req_index
                prompt_list_idx += 1
            else:
                # decode stage
                self.reorder_decode_req_index_list[decode_list_idx] = req_index
                decode_list_idx += 1
        assert decode_list_idx + prompt_list_idx == self.num_reqs

        # Update prompt requests number
        self.num_prompt_req = prompt_list_idx

        reorder_req_num = 0
        for req_index in range(decode_list_idx):
            if self.reorder_decode_req_index_list[req_index] < prompt_list_idx:
                reorder_req_num += 1
            else:
                break

        if reorder_req_num == 0:
            return

        reorder_prompt_list = (
            self.reorder_prompt_req_index_list[:prompt_list_idx]
            [-reorder_req_num:])
        reorder_decode_list = (
            self.reorder_decode_req_index_list[:decode_list_idx]
            [:reorder_req_num])
        assert reorder_decode_list.size == reorder_prompt_list.size

        reorder_prompt_req_index = reorder_prompt_list[0].item()
        cached_req_id = self._req_ids[reorder_prompt_req_index]
        self._req_ids[reorder_prompt_req_index] = None
        cached_token_ids_cpu = self.token_ids_cpu[
            reorder_prompt_req_index].copy()
        cached_num_tokens = self.num_tokens[reorder_prompt_req_index]
        cached_num_tokens_no_spec = self.num_tokens_no_spec[reorder_prompt_req_index]
        cached_num_prompt_tokens_cpu = self.num_prompt_tokens[
            reorder_prompt_req_index]
        cached_num_computed_tokens_cpu = self.num_computed_tokens_cpu[
            reorder_prompt_req_index]
        cached_block_table_cpu = self.block_table.block_table_np[
            reorder_prompt_req_index].copy()
        cached_temperature_cpu = self.temperature_cpu[reorder_prompt_req_index]
        cached_top_p_cpu = self.top_p_cpu[reorder_prompt_req_index]
        cached_top_k_cpu = self.top_k_cpu[reorder_prompt_req_index]
        cached_min_p_cpu = self.min_p_cpu[reorder_prompt_req_index]
        cached_frequency_penalties_cpu = self.frequency_penalties_cpu[
            reorder_prompt_req_index]
        cached_presence_penalties_cpu = self.presence_penalties_cpu[
            reorder_prompt_req_index]
        cached_repetition_penalties_cpu = self.repetition_penalties_cpu[
            reorder_prompt_req_index]
        cached_min_tokens = self.min_tokens.pop(reorder_prompt_req_index, None)
        cached_request_lora_mapping = self.request_lora_mapping[reorder_prompt_req_index]
        cached_generator = self.generators.pop(reorder_prompt_req_index, None)
        cached_logit_bias = self.logit_bias[reorder_prompt_req_index]

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            cached_allowed_token_ids_mask_cpu_tensor = self.allowed_token_ids_mask_cpu_tensor[reorder_prompt_req_index]

        cached_bad_words_token_ids = self.bad_words_token_ids.pop(reorder_prompt_req_index, None)

        for idx in range(reorder_req_num):
            prompt_req_index = reorder_prompt_list[idx].item()
            decode_req_index = reorder_decode_list[idx].item()
            self._move_req(prompt_req_index, decode_req_index)

            prompt_idx = idx + 1
            if prompt_idx != reorder_req_num:
                prompt_req_index = reorder_prompt_list[prompt_idx].item()
                self._move_req(decode_req_index, prompt_req_index)

        reorder_prompt_req_index = reorder_decode_list[-1].item()
        self._req_ids[reorder_prompt_req_index] = cached_req_id
        self.req_id_to_index[cached_req_id] = reorder_prompt_req_index
        self.token_ids_cpu[reorder_prompt_req_index] = cached_token_ids_cpu
        self.num_tokens[reorder_prompt_req_index] = cached_num_tokens
        self.num_tokens_no_spec[reorder_prompt_req_index] = cached_num_tokens_no_spec
        self.num_prompt_tokens[
            reorder_prompt_req_index] = cached_num_prompt_tokens_cpu
        self.num_computed_tokens_cpu[
            reorder_prompt_req_index] = cached_num_computed_tokens_cpu
        self.block_table.block_table_np[reorder_prompt_req_index] = cached_block_table_cpu
        self.temperature_cpu[reorder_prompt_req_index] = cached_temperature_cpu
        self.top_p_cpu[reorder_prompt_req_index] = cached_top_p_cpu
        self.top_k_cpu[reorder_prompt_req_index] = cached_top_k_cpu
        self.min_p_cpu[reorder_prompt_req_index] = cached_min_p_cpu
        self.frequency_penalties_cpu[
            reorder_prompt_req_index] = cached_frequency_penalties_cpu
        self.presence_penalties_cpu[
            reorder_prompt_req_index] = cached_presence_penalties_cpu
        self.repetition_penalties_cpu[
            reorder_prompt_req_index] = cached_repetition_penalties_cpu
        if cached_min_tokens is not None:
            self.min_tokens[reorder_prompt_req_index] = cached_min_tokens
        self.request_lora_mapping[reorder_prompt_req_index] = cached_request_lora_mapping
        if cached_generator is not None:
            self.generators[reorder_prompt_req_index] = cached_generator
        self.logit_bias[reorder_prompt_req_index] = cached_logit_bias

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[reorder_prompt_req_index] = cached_allowed_token_ids_mask_cpu_tensor

        if cached_bad_words_token_ids is not None:
            self.bad_words_token_ids[reorder_prompt_req_index] = cached_bad_words_token_ids