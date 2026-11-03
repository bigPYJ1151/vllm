// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu
//
// Fused MLA decode front end for FP8 block-quantized (W8A16) DeepSeek weights:
// qkv_a_proj, q/kv_a layernorm, q_b_proj, W_UK absorption and RoPE in one op
// (`qkv_proj_with_rope_fused_weight`), plus `mla_decode_fused_cpu`, which adds
// the KV-cache write, the decode attention kernel and the W_UV de-absorption.

// clang-format off

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// [C0, C1] = A @ [B0, B1]
template <typename scalar_t>
void segment_gemm_kernel_impl(
    scalar_t* __restrict__ C0,
    scalar_t* __restrict__ C1,
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B0,
    const at::Float8_e4m3fn* __restrict__ B1,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    scalar_t* __restrict__ Btmp,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t K,
    int64_t block_size_N,
    int64_t block_size_K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB0 = div_up(N0, BLOCK_N);
  const int64_t NB1 = div_up(N1, BLOCK_N);
  const int64_t NB = NB0 + NB1;

  const int64_t scale_size_K = div_up(K, block_size_K);
  const int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  // parallel on [MB, NB0 + NB1]
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int64_t mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    int tid = at::get_thread_num();
    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);

      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = BLOCK_N;

      const at::Float8_e4m3fn* __restrict__ B = nb < NB0 ? B0 : B1;
      const float* __restrict__ Bs = nb < NB0 ? Bs0 : Bs1;
      scalar_t* __restrict__ C = nb < NB0 ? C0 : C1;
      int64_t ldc = nb < NB0 ? N0 : N1;
      int64_t local_nb_start = nb < NB0 ? nb_start : nb_start - N0;
      int64_t new_nb = nb < NB0 ? nb : nb - NB0;

      tinygemm_kernel<scalar_t>(
          /*   A */ A + mb_start * K,
          /*   B */ B + local_nb_start * K /* nb * BLOCK_N * K */,
          /*   C */ C + mb_start * ldc + local_nb_start,
          /* Btmp*/ Btmp + tid * BLOCK_N * K,
          /* Ctmp*/ Ctmp,
          /*Bbias*/ nullptr,
          /*  Bs */ Bs + (new_nb / blocks_n_per_group) * scale_size_K,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ K,
          /* ldb */ nb_size,
          /* ldc */ ldc,
          /* brg */ use_brgemm,
          /* block_size_K */ block_size_K);

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

template <typename scalar_t>
inline float reduce(const scalar_t* __restrict__ x, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  fVec sum_fvec = fVec(float(0));

// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(x + d);
    sum_fvec += x_fvec0 * x_fvec0;
    sum_fvec += x_fvec1 * x_fvec1;
  }
  return vec_reduce_sum(sum_fvec);
}

// map2 from aten functional doesn't have fast bf16->fp32 conversion
template <typename scalar_t>
inline void map2(scalar_t* y, const scalar_t* x, const scalar_t* __restrict__ w, float scale, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  fVec scale_fvec = fVec(scale);

// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(x + d);
    auto [w_fvec0, w_fvec1] = load_float_vec2(w + d);
    x_fvec0 = x_fvec0 * scale_fvec * w_fvec0;
    x_fvec1 = x_fvec1 * scale_fvec * w_fvec1;
    bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
    out_bvec.store(y + d);
  }
}

template <typename scalar_t>
void rms_norm_kernel_impl(
    scalar_t* __restrict__ input0,
    scalar_t* __restrict__ input1,
    const scalar_t* __restrict__ weight0,
    const scalar_t* __restrict__ weight1,
    int64_t M,
    int64_t N0,
    int64_t N1,
    int64_t stride1,
    float eps = 1e-5) {
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      scalar_t* x0 = input0 + m * N0;
      scalar_t* x1 = input1 + m * stride1;
      float scale0 = reduce(x0, N0);
      float scale1 = reduce(x1, N1);
      scale0 = float(1) / std::sqrt(scale0 / N0 + eps);
      scale1 = float(1) / std::sqrt(scale1 / N1 + eps);
      map2(x0, x0, weight0, scale0, N0);
      map2(x1, x1, weight1, scale1, N1);
    }
  });
}

template <typename scalar_t>
inline void rotary(const scalar_t* input, scalar_t* out, const scalar_t* cos, const scalar_t* sin, int64_t size) {
  TORCH_CHECK(false, "rotary scalar path not implemented.");
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void rotary<at::BFloat16>(
    const at::BFloat16* input, at::BFloat16* out, const at::BFloat16* cos, const at::BFloat16* sin, int64_t size) {
  // permute indices
  const __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  const __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  const __m512i idy1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  const __m512i idy2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);

// rotary dim is 64, just 2 iters
#pragma GCC unroll 2
  for (int64_t d = 0; d < size; d += 32) {
    int64_t d2 = d >> 1;
    // load coefs
    __m512 vcos = CVT_BF16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cos + d2)));
    __m512 vsin = CVT_BF16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sin + d2)));
    // load input
    __m512i a16 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + d));
    __m512 a = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
    __m512 b = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
    // from [16, 2] to [2, 16]
    __m512 in1 = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
    __m512 in2 = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
    // out1 = in1 * cos - in2 * sin;
    // out2 = in2 * cos + in1 * sin
    __m512 out1 = _mm512_sub_ps(_mm512_mul_ps(in1, vcos), _mm512_mul_ps(in2, vsin));
    __m512 out2 = _mm512_add_ps(_mm512_mul_ps(in2, vcos), _mm512_mul_ps(in1, vsin));
    // from [2, 16] to [16, 2]
    a = _mm512_mask_permutex2var_ps(out1, 0xffff, idy1, out2);
    b = _mm512_mask_permutex2var_ps(out1, 0xffff, idy2, out2);

    _mm512_storeu_si512(reinterpret_cast<__m512i*>((out + d)), (__m512i)(_mm512_cvtne2ps_pbh(b, a)));
  }
}
#endif

template <typename scalar_t>
void rotary_emb_kernel_impl(
    scalar_t* q_pe_out,
    scalar_t* k_pe_out,
    const scalar_t* q_pe,
    const scalar_t* k_pe,
    const int64_t* pos,
    const scalar_t* cos_sin,
    int64_t num_seqs,
    int64_t num_heads,
    int64_t rotary_dim,
    int64_t q_strideB,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t oq_strideB,
    int64_t oq_strideH,
    int64_t ok_strideB) {
  TORCH_CHECK(rotary_dim % 32 == 0, "rotary_dim is not 32x.");
  const int64_t rotary_offset = rotary_dim / 2;

  // parallel on [num_seqs, num_heads + 1]
  // top [num_heads] handle q_pe and bottom [1] handle k_pe
  at::parallel_for(0, num_seqs * (num_heads + 1), GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, num_seqs, head_id, num_heads + 1);

    for (int64_t i = begin; i < end; ++i) {
      UNUSED(i);
      // get cos and sin cache ptr
      int64_t index = pos[seq];
      const scalar_t* cos = cos_sin + index * rotary_dim;
      const scalar_t* sin = cos + rotary_offset;

      const scalar_t* input =
          (head_id < num_heads) ? q_pe + seq * q_strideB + head_id * q_strideH : k_pe + seq * k_strideB;
      scalar_t* out =
          (head_id < num_heads) ? q_pe_out + seq * oq_strideB + head_id * oq_strideH : k_pe_out + seq * ok_strideB;
      rotary<scalar_t>(input, out, cos, sin, rotary_dim);

      // move to the next index
      data_index_step(seq, num_seqs, head_id, num_heads + 1);
    }
  });
}

}  // anonymous namespace

extern void
bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni, const std::optional<at::Tensor>& scale);

extern at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

extern void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    const std::optional<at::Tensor>& key,
    const std::optional<at::Tensor>& value,
    const std::optional<at::Tensor>& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap,
    bool is_cross_attn,
    int64_t sliding_window_size,
    std::optional<at::Tensor> encoder_lens,
    std::optional<at::Tensor> sinks);

// NB: shapes in DeepSeek R1 (per TP4 rank)
//
//   hidden_states         : [num_seqs, hidden_size]                 [1, 7168]
//   qkv_a_proj_weight     : [q_lora_rank + kv_lora_rank + rope, K]  [2112, 7168], FP8 VNNI-packed
//   q_b_proj_weight       : [num_heads * qk_head_dim, q_lora_rank]  [6144, 1536], FP8 VNNI-packed
//   w_kc                  : [num_heads, kv_lora_rank, qk_nope]      [32, 512, 128]
//   returns q_input [num_seqs, num_heads, kv_lora_rank + rope], k_input [num_seqs, 1, kv_lora_rank + rope], v_input
std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope_fused_weight(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> qkv_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> w_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim) {
  TORCH_CHECK(use_fp8_w8a16 && !use_int8_w8a8, "qkv_proj_with_rope_fused_weight: only FP8 W8A16 is supported.");
  TORCH_CHECK(is_vnni, "qkv_proj_with_rope_fused_weight: expect weights are prepacked!");
  TORCH_CHECK(qkv_a_proj_scale.has_value() && q_b_proj_scale.has_value(), "missing FP8 weight scales.");
  TORCH_CHECK(block_size.has_value() && block_size.value().size() == 2, "block_size should be 2D for fp8 w8a16.");

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(positions);
  CHECK_INPUT(cos_sin_cache);
  CHECK_EQ(q_a_layernorm_weight.scalar_type(), st);
  CHECK_EQ(kv_a_layernorm_weight.scalar_type(), st);
  CHECK_EQ(positions.scalar_type(), at::kLong);
  CHECK_EQ(cos_sin_cache.scalar_type(), st);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(3, w_kc);
  CHECK_DIM(1, q_a_layernorm_weight);
  CHECK_DIM(1, kv_a_layernorm_weight);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, cos_sin_cache);

  const int64_t num_seqs = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t num_heads = w_kc.size(0);
  const int64_t qk_head_dim = q_b_proj_weight.size(0) / num_heads;
  const int64_t qk_nope_head_dim = w_kc.size(2);
  const int64_t rotary_dim = cos_sin_cache.size(1);
  const int64_t kv_dim = kv_lora_rank + qk_rope_head_dim;
  const int64_t block_size_N = block_size.value()[0];
  const int64_t block_size_K = block_size.value()[1];

  CHECK_EQ(qkv_a_proj_weight.size(0), q_lora_rank + kv_dim);
  CHECK_EQ(qkv_a_proj_weight.size(1), hidden_size);
  CHECK_EQ(w_kc.size(1), kv_lora_rank);
  CHECK_EQ(positions.numel(), num_seqs);
  CHECK_EQ(rotary_dim, qk_rope_head_dim);
  CHECK_EQ(q_a_layernorm_weight.numel(), q_lora_rank);
  CHECK_EQ(kv_a_layernorm_weight.numel(), kv_lora_rank);
  CHECK_EQ(q_b_proj_weight.size(1), q_lora_rank);

  // q_a_proj and kv_a_proj_with_mqa are row blocks of the fused weight and its scale
  auto weight_chunks = at::split_with_sizes(qkv_a_proj_weight, {q_lora_rank, kv_dim}, 0);
  auto scale_chunks = at::split_with_sizes(
      qkv_a_proj_scale.value(), {div_up(q_lora_rank, block_size_N), div_up(kv_dim, block_size_N)}, 0);
  at::Tensor q_a_proj_weight = weight_chunks[0];
  at::Tensor kv_a_proj_weight = weight_chunks[1];
  at::Tensor q_a_proj_s = scale_chunks[0];
  at::Tensor kv_a_proj_s = scale_chunks[1];
  CHECK_EQ(q_a_proj_s.size(1), div_up(hidden_size, block_size_K));
  CHECK_EQ(kv_a_proj_s.size(1), div_up(hidden_size, block_size_K));

  const auto options = hidden_states.options();
  auto q_input = at::empty({num_seqs, num_heads, kv_dim}, options);
  auto k_input = at::empty({num_seqs, 1, kv_dim}, options);
  auto v_input = k_input.narrow(-1, 0, kv_lora_rank);
  auto qa = at::empty({num_seqs, q_lora_rank}, options);

  // stage 1: q_a_proj and kv_a_proj
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "qkv_proj_kernel_impl", [&] {
    const int BLOCK_N = block_size_n();
    const int num_threads = at::get_num_threads();
    auto buffer = at::empty({num_threads, BLOCK_N * hidden_size}, options);
    segment_gemm_kernel_impl<scalar_t>(
        qa.data_ptr<scalar_t>(),
        k_input.data_ptr<scalar_t>(),
        hidden_states.data_ptr<scalar_t>(),
        q_a_proj_weight.data_ptr<at::Float8_e4m3fn>(),
        kv_a_proj_weight.data_ptr<at::Float8_e4m3fn>(),
        q_a_proj_s.data_ptr<float>(),
        kv_a_proj_s.data_ptr<float>(),
        buffer.data_ptr<scalar_t>(),
        num_seqs,
        q_lora_rank,
        kv_dim,
        hidden_size,
        block_size_N,
        block_size_K);
  });

  // stage 2: apply rmsnorm inplace
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "rms_norm_kernel_impl", [&] {
    rms_norm_kernel_impl<scalar_t>(
        qa.data_ptr<scalar_t>(),
        v_input.data_ptr<scalar_t>(),
        q_a_layernorm_weight.data_ptr<scalar_t>(),
        kv_a_layernorm_weight.data_ptr<scalar_t>(),
        num_seqs,
        q_lora_rank,
        kv_lora_rank,
        kv_dim,
        eps);
  });

  // stage 3: q_b_proj
  std::optional<at::Tensor> bias;
  at::Tensor qb = fp8_scaled_mm_cpu(qa, q_b_proj_weight, q_b_proj_scale.value(), block_size.value(), bias, st, is_vnni);
  qb.as_strided_({num_seqs, num_heads, qk_head_dim}, {num_heads * qk_head_dim, qk_head_dim, 1});

  // stage 4: absorb W_UK into the nope part of the query
  auto q_nope = qb.narrow(2, 0, qk_nope_head_dim).transpose_(0, 1);
  auto q_nope_out = q_input.narrow(2, 0, kv_lora_rank).transpose_(0, 1);
  bmm_cpu(q_nope_out, q_nope, w_kc, is_vnni, w_scale);

  // stage 5: rope
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "rotary_emb_kernel_impl", [&] {
    rotary_emb_kernel_impl<scalar_t>(
        q_input.data_ptr<scalar_t>() + kv_lora_rank,
        k_input.data_ptr<scalar_t>() + kv_lora_rank,
        qb.data_ptr<scalar_t>() + qk_nope_head_dim,
        k_input.data_ptr<scalar_t>() + kv_lora_rank,
        positions.data_ptr<int64_t>(),
        cos_sin_cache.data_ptr<scalar_t>(),
        num_seqs,
        num_heads,
        rotary_dim,
        num_heads * qk_head_dim,
        qk_head_dim,
        kv_dim,
        num_heads * kv_dim,
        kv_dim,
        kv_dim);
  });

  return std::make_tuple(q_input, k_input, v_input);
}

// MLA decode for the single-token batch: fused qkv/rope front end, KV-cache write,
// decode attention and W_UV de-absorption. `out` is [num_seqs, num_heads * v_head_dim].
void mla_decode_fused_cpu(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& w_vc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    at::Tensor& qkv_a_proj_scale,
    at::Tensor& q_b_proj_scale,
    std::optional<at::Tensor> w_scale,
    std::vector<int64_t> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim,
    at::Tensor& kv_cache,
    at::Tensor& slot_mapping,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    int64_t num_kv_splits,
    double sm_scale,
    at::Tensor& out) {
  const int64_t num_seqs = hidden_states.size(0);
  const int64_t num_heads = w_kc.size(0);
  const int64_t v_head_dim = w_vc.size(1);
  const int64_t kv_dim = kv_lora_rank + qk_rope_head_dim;
  CHECK_EQ(slot_mapping.numel(), num_seqs);
  CHECK_EQ(out.size(0), num_seqs);
  CHECK_EQ(out.size(1), num_heads * v_head_dim);
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "slot_mapping must be int64.");

  auto [q_input, k_input, v_input] = qkv_proj_with_rope_fused_weight(
      hidden_states,
      qkv_a_proj_weight,
      q_b_proj_weight,
      w_kc,
      q_a_layernorm_weight,
      kv_a_layernorm_weight,
      positions,
      cos_sin_cache,
      eps,
      /*use_int8_w8a8=*/false,
      /*use_fp8_w8a16=*/true,
      qkv_a_proj_scale,
      q_b_proj_scale,
      w_scale,
      /*is_vnni=*/true,
      block_size,
      q_lora_rank,
      kv_lora_rank,
      qk_rope_head_dim);

  // KV-cache write: rows of [kv_c_normed | rotated k_pe] go to their slots
  auto kv_flat = kv_cache.view({-1, 1, kv_dim});
  CHECK_EQ(kv_flat.scalar_type(), k_input.scalar_type());
  const size_t row_bytes = kv_dim * k_input.element_size();
  const char* src = static_cast<const char*>(k_input.data_ptr());
  char* dst = static_cast<char*>(kv_flat.data_ptr());
  const int64_t* slots = slot_mapping.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_seqs; ++i) {
    if (slots[i] >= 0) {
      std::memcpy(dst + slots[i] * row_bytes, src + i * row_bytes, row_bytes);
    }
  }

  auto v_buffer = kv_flat.narrow(-1, 0, kv_lora_rank);
  // the kernel writes every element of both buffers before reading it
  auto attn_logits =
      at::empty({num_seqs, num_heads, num_kv_splits, kv_lora_rank + 1}, q_input.options().dtype(at::kFloat));
  auto o = at::empty({num_seqs, num_heads, kv_lora_rank}, q_input.options());
  decode_attention_cpu(
      q_input,
      kv_flat,
      v_buffer,
      o,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      attn_logits,
      req_to_token,
      req_pool_indices,
      seq_lens,
      sm_scale,
      0.0,
      false,
      0,
      std::nullopt,
      std::nullopt);

  auto out_view = out.view({num_seqs, num_heads, v_head_dim}).transpose(0, 1);
  auto o_t = o.transpose(0, 1);
  bmm_cpu(out_view, o_t, w_vc, /*is_vnni=*/true, w_scale);
}
