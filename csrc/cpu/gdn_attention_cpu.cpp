// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused CPU forward of the Gated DeltaNet (GDN) core for non-speculative
// batches: causal conv1d, q/k/v split, gating and the delta-rule recurrence
// for the decode and prefill parts of one batch, in one op. The compute is
// done by the existing conv / fla kernels; this file only removes the
// per-layer Python orchestration.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

at::Tensor causal_conv1d_fwd_cpu(
    const at::Tensor& x, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state, bool silu_activation,
    int64_t pad_slot_id, bool is_weight_packed);

at::Tensor causal_conv1d_update_cpu(
    const at::Tensor& x, const at::Tensor& conv_states,
    const at::Tensor& weight, const std::optional<at::Tensor>& bias,
    bool silu_activation, const std::optional<at::Tensor>& num_accepted_tokens,
    const std::optional<at::Tensor>& conv_state_indices, int64_t pad_slot_id,
    bool is_weight_packed);

at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
    const at::Tensor& A_log, const at::Tensor& dt_bias, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& a,
    const at::Tensor& b, at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices, const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel, double softplus_beta, double softplus_threshold);

std::tuple<at::Tensor, at::Tensor> fused_gdn_gating_cpu(
    const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& dt_bias);

std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& g, const at::Tensor& beta,
    const at::Tensor& initial_state, bool output_final_state,
    const at::Tensor& cu_seqlens, bool head_first, bool use_qk_l2norm_in_kernel,
    const at::Tensor& initial_state_indices, double eps);

namespace {

constexpr int64_t kPadSlotId = -1;

// Splits [T, 2 * H * D + Hv * Dv] into (query, key, value) of shape
// (1, T, H, D) / (1, T, H, D) / (1, T, Hv, Dv). Views only.
std::tuple<at::Tensor, at::Tensor, at::Tensor> split_qkv_views(
    const at::Tensor& qkv, int64_t head_k_dim, int64_t head_v_dim,
    int64_t num_v_heads) {
  const int64_t T = qkv.size(0);
  const int64_t v_width = num_v_heads * head_v_dim;
  const int64_t k_width = (qkv.size(1) - v_width) / 2;
  const int64_t num_k_heads = k_width / head_k_dim;
  auto q = qkv.narrow(1, 0, k_width).view({1, T, num_k_heads, head_k_dim});
  auto k = qkv.narrow(1, k_width, k_width).view({1, T, num_k_heads, head_k_dim});
  auto v = qkv.narrow(1, 2 * k_width, v_width)
               .view({1, T, num_v_heads, head_v_dim});
  return {q, k, v};
}

}  // namespace

// mixed_qkv: [num_tokens, 2 * H * Dk + Hv * Dv], a / b: [num_tokens, Hv]
// conv_state: [slots, state_len = width - 1, dim] (SD layout)
// conv_weight: packed causal conv1d weight
// ssm_state: [slots, Hv, Dv, Dk]
// state_indices / query_start_loc / has_initial_state: non-spec metadata,
// decode requests first, then prefill requests.
namespace {

void gdn_attention_core_impl(
    const at::Tensor& mixed_qkv_in, const at::Tensor& b_in,
    const at::Tensor& a_in, at::Tensor& core_attn_out, at::Tensor& conv_state,
    const at::Tensor& conv_weight, const std::optional<at::Tensor>& conv_bias,
    at::Tensor& ssm_state_in, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const at::Tensor& state_indices,
    const at::Tensor& query_start_loc,
    const std::optional<at::Tensor>& has_initial_state, int64_t num_decodes,
    int64_t num_decode_tokens, int64_t num_prefills,
    int64_t num_prefill_tokens, int64_t head_k_dim, int64_t head_v_dim,
    bool silu) {
  TORCH_CHECK(mixed_qkv_in.scalar_type() == at::kBFloat16,
              "gdn_attention_core_cpu: only BF16 is supported");
  const at::Tensor mixed_qkv = mixed_qkv_in.contiguous();
  const at::Tensor a = a_in.contiguous();
  const at::Tensor b = b_in.contiguous();

  const at::Tensor conv_state_dim_first = conv_state.transpose(1, 2);
  const int64_t num_slots = ssm_state_in.size(0);
  const int64_t num_v_heads = ssm_state_in.size(1);
  at::Tensor ssm_state =
      ssm_state_in.view({num_slots, num_v_heads, head_k_dim, head_v_dim});

  if (num_decodes > 0) {
    const auto decode_indices = state_indices.narrow(0, 0, num_decodes);
    const at::Tensor conv_out = causal_conv1d_update_cpu(
        mixed_qkv.narrow(0, 0, num_decode_tokens), conv_state_dim_first,
        conv_weight, conv_bias, silu, std::nullopt, decode_indices, kPadSlotId,
        /*is_weight_packed=*/true);
    auto [q, k, v] =
        split_qkv_views(conv_out, head_k_dim, head_v_dim, num_v_heads);
    const at::Tensor attn_out = fused_sigmoid_gating_delta_rule_update_cpu(
        A_log, dt_bias, q, k, v, a.narrow(0, 0, num_decode_tokens),
        b.narrow(0, 0, num_decode_tokens), ssm_state, decode_indices,
        query_start_loc.narrow(0, 0, num_decodes + 1),
        /*use_qk_l2norm_in_kernel=*/true, 1.0, 20.0);
    core_attn_out.narrow(0, 0, num_decode_tokens).copy_(attn_out.squeeze(1));
  }

  if (num_prefills > 0) {
    TORCH_CHECK(has_initial_state.has_value(),
                "gdn_attention_core_cpu: has_initial_state is required for "
                "prefill requests");
    const auto prefill_indices =
        state_indices.narrow(0, num_decodes, num_prefills);
    const auto prefill_start_loc =
        query_start_loc.narrow(0, num_decodes, num_prefills + 1) -
        num_decode_tokens;
    const auto prefill_has_initial_state =
        has_initial_state->narrow(0, num_decodes, num_prefills);

    const at::Tensor conv_out =
        causal_conv1d_fwd_cpu(
            mixed_qkv.narrow(0, num_decode_tokens, num_prefill_tokens)
                .transpose(0, 1),
            conv_weight, conv_bias, conv_state_dim_first, prefill_start_loc,
            prefill_indices, prefill_has_initial_state, silu, kPadSlotId,
            /*is_weight_packed=*/true)
            .transpose(0, 1)
            .contiguous();
    auto [q, k, v] =
        split_qkv_views(conv_out, head_k_dim, head_v_dim, num_v_heads);
    auto [g, beta] = fused_gdn_gating_cpu(
        A_log, a.narrow(0, num_decode_tokens, num_prefill_tokens),
        b.narrow(0, num_decode_tokens, num_prefill_tokens), dt_bias);

    const at::Tensor no_initial_state =
        prefill_indices.index({~prefill_has_initial_state});
    if (no_initial_state.numel() > 0) {
      ssm_state.index_fill_(0, no_initial_state.to(at::kLong), 0.0);
    }
    auto [attn_out, final_state] = chunk_gated_delta_rule_cpu(
        q.contiguous(), k.contiguous(), v.contiguous(), g, beta, ssm_state,
        /*output_final_state=*/true, prefill_start_loc, /*head_first=*/false,
        /*use_qk_l2norm_in_kernel=*/true, prefill_indices, 1e-5);
    core_attn_out.narrow(0, num_decode_tokens, num_prefill_tokens)
        .copy_(attn_out.squeeze(0));
  }
}

}  // namespace

void gdn_attention_core_cpu(
    const at::Tensor& mixed_qkv, const at::Tensor& b, const at::Tensor& a,
    at::Tensor& core_attn_out, at::Tensor& conv_state,
    const at::Tensor& conv_weight, const std::optional<at::Tensor>& conv_bias,
    at::Tensor& ssm_state, const at::Tensor& A_log, const at::Tensor& dt_bias,
    const at::Tensor& state_indices, const at::Tensor& query_start_loc,
    const std::optional<at::Tensor>& has_initial_state, int64_t num_decodes,
    int64_t num_decode_tokens, int64_t num_prefills,
    int64_t num_prefill_tokens, int64_t head_k_dim, int64_t head_v_dim,
    bool silu) {
  gdn_attention_core_impl(mixed_qkv, b, a, core_attn_out, conv_state,
                          conv_weight, conv_bias, ssm_state, A_log, dt_bias,
                          state_indices, query_start_loc, has_initial_state,
                          num_decodes, num_decode_tokens, num_prefills,
                          num_prefill_tokens, head_k_dim, head_v_dim, silu);
}

// Same as gdn_attention_core_cpu, but the request counts come from the int32
// tensor [num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens]
// and has_initial_state is empty when there is no prefill. All inputs are
// tensors, so the op can be called straight from compiled code. A step
// without requests is a no-op.
void gdn_attention_core_direct_cpu(
    const at::Tensor& mixed_qkv, const at::Tensor& b, const at::Tensor& a,
    at::Tensor& core_attn_out, at::Tensor& conv_state,
    const at::Tensor& conv_weight, const std::optional<at::Tensor>& conv_bias,
    at::Tensor& ssm_state, const at::Tensor& A_log, const at::Tensor& dt_bias,
    const at::Tensor& state_indices, const at::Tensor& query_start_loc,
    const at::Tensor& has_initial_state, const at::Tensor& counts,
    int64_t head_k_dim, int64_t head_v_dim, bool silu) {
  const int32_t* count = counts.data_ptr<int32_t>();
  if (count[0] + count[2] == 0) {
    return;
  }
  gdn_attention_core_impl(mixed_qkv, b, a, core_attn_out, conv_state,
                          conv_weight, conv_bias, ssm_state, A_log, dt_bias,
                          state_indices, query_start_loc, has_initial_state,
                          count[0], count[1], count[2], count[3], head_k_dim,
                          head_v_dim, silu);
}
