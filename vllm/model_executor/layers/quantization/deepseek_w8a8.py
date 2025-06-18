# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               )
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import choose_scaled_mm_linear_kernel
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import ScaledMMLinearLayerConfig
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

logger = init_logger(__name__)

class DeepSeekW8A8Config(QuantizationConfig):
    """Config class for Int8 experts quantization."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_w8a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepSeekW8A8Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return DeepSeekW8A8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return DeepSeekW8A8MoEMethod(self)
        return None

class DeepSeekW8A8LinearMethod(LinearMethodBase):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quantization_config: DeepSeekW8A8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes

        scaled_mm_linear_kernel_config = ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True)

        kernel_type = choose_scaled_mm_linear_kernel(
            scaled_mm_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for DeepSeekW8A8 Linear",
                        kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1),
                                dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(c=scaled_mm_linear_kernel_config,
                                  w_q_param_name="weight",
                                  w_s_param_name="weight_scale",
                                  i_s_param_name="input_scale",
                                  i_zp_param_name="input_zero_point",
                                  azp_adj_param_name="azp_adj")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        if self.kernel is None:
            raise ValueError("A kernel must be defined for each layer")
        return self.kernel.apply_weights(layer, x, bias=bias)


class DeepSeekW8A8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        assert current_platform.is_cpu() and current_platform.get_cpu_architecture() == CpuArchEnum.X86
        assert envs.VLLM_CPU_SGL_KERNEL and torch._C._cpu._is_amx_tile_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.w13_weight = w13_weight
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        layer.w2_weight = w2_weight
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.w13_input_scale = w13_input_scale

        w2_input_scale = None
        layer.w2_input_scale = w2_input_scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe import cpu_fused_moe
        packed_w13_weight = torch.ops._C.convert_weight_packed(layer.w13_weight)    
        layer.w13_weight = torch.nn.Parameter(packed_w13_weight, requires_grad=False)
        packed_w2_weight = torch.ops._C.convert_weight_packed(layer.w2_weight)    
        layer.w2_weight = torch.nn.Parameter(packed_w2_weight, requires_grad=False)
        layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        return layer.cpu_fused_moe(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
        )
