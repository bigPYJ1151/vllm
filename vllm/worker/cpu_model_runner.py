import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, PromptAdapterConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class CPUModelInput(ModelRunnerInputBase):
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
            cls: Type["CPUModelInput"],
            tensor_dict: Dict[str, Any],
            attn_backend: Optional["AttentionBackend"] = None
    ) -> "CPUModelInput":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class CPUModelInputBuilder(ModelRunnerInputBuilderBase[CPUModelInput]):

    class ModelInputData:

        def __init__(self):
            self.input_tokens = []
            self.input_positions = []
            self.seq_lens = []
            self.query_lens = []
            self.prefill_block_tables = []
            self.decode_block_tables = []
            self.max_decode_seq_len = 0
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.num_decode_tokens = 0
            self.slot_mapping = []
            self.multi_modal_inputs_list = []

    def __init__(self, runner: "CPUModelRunner"):
        super().__init__()

        self.runner = runner
        self.chunked_prefill = runner.scheduler_config.chunked_prefill_enabled

        self.input_data = CPUModelInputBuilder.ModelInputData()

    def _compute_input_tokens(self, data: ModelInputData,
                              seq_group_metadata: SequenceGroupMetadata,
                              id: int):
        """
        Compute input tokens, positions, block table and slot mapping.
        """
        is_prompt = seq_group_metadata.is_prompt
        token_chunk_size = seq_group_metadata.token_chunk_size
        block_size = self.runner.block_size

        seq_data = seq_group_metadata.seq_data[id]
        block_table = seq_group_metadata.block_tables[id]
        seq_len = seq_data.get_len()

        if is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = context_len + token_chunk_size
            
            # For prefix caching
            prefix_cache_block_num = len(seq_group_metadata.computed_block_nums)
            if prefix_cache_block_num > 0:
                prefix_cache_len = prefix_cache_block_num * self.runner.block_size
                self.chunked_prefill = True
                if prefix_cache_len <= context_len:
                    # We already passed the cache hit region,
                    # so do normal computation.
                    pass
                elif context_len < prefix_cache_len < seq_len:
                    # Partial hit. Compute the missing part.
                    context_len = prefix_cache_len
                    token_chunk_size = seq_len - context_len
                elif seq_len <= prefix_cache_len:
                    # Full hit. Only compute the last token to avoid
                    # erroneous behavior. FIXME: Ideally we should directly
                    # mark all tokens as computed in the scheduler and do not
                    # schedule this sequence, so this case should not happen.
                    context_len = seq_len - 1
                    token_chunk_size = 1

            tokens = seq_data.get_token_ids()
            if context_len != 0 or seq_len < len(tokens):
                # For chunked prefill
                tokens = tokens[context_len:seq_len]

            token_positions = range(context_len, seq_len)

            slot_mapping = [-1] * len(token_positions)
            for i, pos in enumerate(token_positions):
                block_number = block_table[pos // block_size]
                block_offset = pos % block_size
                slot = block_number * block_size + block_offset
                slot_mapping[i] = slot

            # Update fields
            data.input_tokens.extend(tokens)
            data.input_positions.extend(token_positions)
            data.num_prefills += 1
            data.num_prefill_tokens += len(tokens)
            data.slot_mapping.extend(slot_mapping)
            data.query_lens.append(len(tokens))
            data.prefill_block_tables.append(block_table)
        else:
            tokens = seq_data.get_last_token_id()
            token_positions = seq_len - 1

            block_number = block_table[token_positions // block_size]
            block_offset = token_positions % block_size
            slot = block_number * block_size + block_offset

            # For paged_attention kernel
            if self.runner.sliding_window:
                start_idx = max(0, seq_len - self.runner.sliding_window)
                start_block = start_idx // block_size
                start_idx = start_block * block_size
                seq_len = seq_len - start_idx
                block_table = block_table[start_block:]

            # Update fields
            data.input_tokens.append(tokens)
            data.input_positions.append(token_positions)
            data.max_decode_seq_len = max(data.max_decode_seq_len, seq_len)
            data.num_decode_tokens += 1
            data.slot_mapping.append(slot)
            data.decode_block_tables.append(block_table)
            data.query_lens.append(1)

        data.seq_lens.append(seq_len)

    def _compute_mm_data(self, data: ModelInputData,
                         seq_group_metadata: SequenceGroupMetadata, id: int):
        mm_data = seq_group_metadata.multi_modal_data
        if mm_data:
            mm_kwargs = self.runner.multi_modal_input_mapper(mm_data)
            data.multi_modal_inputs_list.append(mm_kwargs)

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert len(seq_ids) == 1
            self._compute_mm_data(self.input_data, seq_group_metadata, 0)

        for id in seq_ids:
            self._compute_input_tokens(self.input_data, seq_group_metadata, id)

    def build(self) -> CPUModelInput:
        input_data = self.input_data
        prefill_seq_lens = input_data.seq_lens[0:input_data.num_prefills]
        prefill_query_lens = input_data.query_lens[0:input_data.num_prefills]
        input_tokens = torch.tensor(input_data.input_tokens,
                                    dtype=torch.long,
                                    device="cpu")
        input_positions = torch.tensor(input_data.input_positions,
                                       dtype=torch.long,
                                       device="cpu")
        slot_mapping = torch.tensor(input_data.slot_mapping,
                                    dtype=torch.long,
                                    device="cpu")

        if self.chunked_prefill and input_data.num_prefill_tokens != 0:
            prefill_block_tables = make_tensor_with_pad(
                self.input_data.prefill_block_tables,
                pad=0,
                dtype=torch.int,
                device="cpu",
            )
            query_lens_tensor = torch.tensor(prefill_query_lens,
                                             dtype=torch.int32,
                                             device="cpu")
            kv_lens_tensor = torch.tensor(prefill_seq_lens,
                                          dtype=torch.int32,
                                          device="cpu")
            query_start_loc = torch.zeros(input_data.num_prefills + 1,
                                          dtype=torch.int32,
                                          device="cpu")
            kv_start_loc = torch.zeros(input_data.num_prefills + 1,
                                       dtype=torch.int32,
                                       device="cpu")
            torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=query_start_loc[1:])
            torch.cumsum(kv_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=kv_start_loc[1:])
            max_query_len = max(prefill_query_lens)
            max_kv_len = max(prefill_seq_lens)
        else:
            prefill_block_tables = None
            query_start_loc = None
            kv_start_loc = None
            max_query_len = None
            max_kv_len = None

        if input_data.num_decode_tokens != 0:
            seq_lens_tensor = torch.tensor(
                input_data.seq_lens[input_data.num_prefills:],
                dtype=torch.int,
                device="cpu",
            )
            block_tables = make_tensor_with_pad(
                self.input_data.decode_block_tables,
                pad=0,
                dtype=torch.int,
                device="cpu",
            )
        else:
            block_tables = None
            seq_lens_tensor = None

        attn_metadata = self.runner.attn_backend.make_metadata(
            chunked_prefill=self.chunked_prefill,
            seq_lens=prefill_seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_kv_len=max_kv_len,
            query_start_loc=query_start_loc,
            kv_start_loc=kv_start_loc,
            max_decode_seq_len=input_data.max_decode_seq_len,
            num_prefills=input_data.num_prefills,
            num_prefill_tokens=input_data.num_prefill_tokens,
            num_decode_tokens=input_data.num_decode_tokens,
            block_tables=block_tables,
            prefill_block_tables=prefill_block_tables,
            slot_mapping=slot_mapping)

        multi_modal_kwargs = MultiModalInputs.batch(
            input_data.multi_modal_inputs_list)

        return CPUModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            seq_lens=input_data.seq_lens,
            query_lens=input_data.query_lens,
            attn_metadata=attn_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
        )


class CPUModelRunner(ModelRunnerBase[CPUModelInput]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.prompt_adapter_config = prompt_adapter_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model

    def load_model(self) -> None:
        self.model = get_model(model_config=self.model_config,
                               load_config=self.load_config,
                               device_config=self.device_config,
                               lora_config=self.lora_config,
                               parallel_config=self.parallel_config,
                               scheduler_config=self.scheduler_config,
                               cache_config=self.cache_config)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> CPUModelInput:
        return CPUModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> CPUModelInput:
        builder = CPUModelInputBuilder(self)

        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        model_input = builder.build()

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            pin_memory=False,
            generators=self.get_generators(finished_requests_ids))
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: CPUModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        model_executable = self.model
        hidden_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs,
                                         device=self.device),
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]
