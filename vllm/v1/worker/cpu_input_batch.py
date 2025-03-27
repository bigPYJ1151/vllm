# SPDX-License-Identifier: Apache-2.0
from typing import Optional, cast

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
