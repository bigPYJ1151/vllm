// Fused gate_up -> silu_and_mul -> down for FP8 block-quantized MLPs (DeepSeek
// shared experts and dense layers). Few tokens go through the fused W8A16
// `shared_expert_cpu` kernel; larger batches keep the W8A8 AMX GEMMs.

// clang-format off

#include "common.h"

extern at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    const std::optional<at::Tensor>& fused_experts_out,
    const std::optional<double> routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni);

extern at::Tensor fp8_scaled_mm_with_quant(
    const at::Tensor& act,
    const std::optional<at::Tensor>& act_scales,
    bool channelwise,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype);

// x: [M, K]; *_w8a16 are the VNNI-packed weights with block scales, *_w8a8 the
// AMX-FP8 prepacked weights with expanded scales. Returns the un-reduced [M, K] output.
at::Tensor mlp_fp8_fused_cpu(
    at::Tensor& x,
    at::Tensor& w1_w8a16,
    at::Tensor& s1_w8a16,
    at::Tensor& w2_w8a16,
    at::Tensor& s2_w8a16,
    at::Tensor& w1_w8a8,
    at::Tensor& s1_w8a8,
    at::Tensor& w2_w8a8,
    at::Tensor& s2_w8a8,
    int64_t max_w8a16_tokens) {
  CHECK_DIM(2, x);
  auto x_c = x.contiguous();
  if (x_c.size(0) <= max_w8a16_tokens) {
    return shared_expert_cpu(
        x_c,
        w1_w8a16,
        w2_w8a16,
        std::nullopt,
        std::nullopt,
        /*inplace=*/false,
        /*use_int8_w8a8=*/false,
        /*use_fp8_w8a16=*/true,
        s1_w8a16,
        s2_w8a16,
        std::vector<int64_t>{128, 128},
        /*is_vnni=*/true);
  }
  auto gate_up =
      fp8_scaled_mm_with_quant(x_c, std::nullopt, /*channelwise=*/true, w1_w8a8, s1_w8a8, std::nullopt, x_c.scalar_type());
  const int64_t n = gate_up.size(-1) / 2;
  auto act = at::silu(gate_up.narrow(-1, 0, n)).mul_(gate_up.narrow(-1, n, n));
  return fp8_scaled_mm_with_quant(act, std::nullopt, /*channelwise=*/true, w2_w8a8, s2_w8a8, std::nullopt, x_c.scalar_type());
}

extern at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// Bias-free FP8 linear: the W8A16 kernel up to `max_w8a16_tokens` tokens, the W8A8 AMX one above.
at::Tensor fp8_linear_small_m_cpu(
    at::Tensor& x,
    at::Tensor& w_w8a16,
    at::Tensor& s_w8a16,
    at::Tensor& w_w8a8,
    at::Tensor& s_w8a8,
    int64_t max_w8a16_tokens) {
  CHECK_DIM(2, x);
  auto x_c = x.contiguous();
  if (x_c.size(0) <= max_w8a16_tokens) {
    return fp8_scaled_mm_cpu(x_c, w_w8a16, s_w8a16, std::vector<int64_t>{128, 128}, std::nullopt, x_c.scalar_type(), true);
  }
  return fp8_scaled_mm_with_quant(x_c, std::nullopt, /*channelwise=*/true, w_w8a8, s_w8a8, std::nullopt, x_c.scalar_type());
}
