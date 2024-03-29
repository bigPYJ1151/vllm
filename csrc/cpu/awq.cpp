#include "cpu_types.hpp"

namespace {
template <typename scalar_t> struct KernelVecType {
  using scalar_vec_t = void;
  using dot_vec_t = void;
};

template <> struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
  using dot_vec_t = vec_op::FP32Vec16;
};

template <> struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
  using dot_vec_t = vec_op::FP32Vec16;
};

FORCE_INLINE vec_op::FP32Vec16
cast_packed_s4x16_to_fp32x16(const int *__restrict__ packed_weight_ptr,
                             const __m512 table, const __m512i shift_offset) {
  __m256i packed1_vec = _mm256_set1_epi32(*packed_weight_ptr);
  __m256i packed2_vec = _mm256_set1_epi32(*(packed_weight_ptr + 1));
  __m512i packed_vec = (__m512i)_mm512_insertf32x8(
      (__m512)_mm512_castsi256_si512(packed1_vec), (__m256)packed2_vec, 1);
  packed_vec = _mm512_srlv_epi32(packed_vec, shift_offset);
  __m512 f32_packed_vec = _mm512_permutexvar_ps(packed_vec, table);
  return vec_op::FP32Vec16(f32_packed_vec);
}

template <typename scalar_t>
void awq_dequantize_impl(
    const int *__restrict__ packed_weight, // [in_c, qout_c]
    const scalar_t *__restrict__ scale,    // [in_c / group_size, out_c]
    const int *__restrict__ zeros,         // [in_c / group_size, qout_c]
    scalar_t *__restrict__ weight,         // [in_c, out_c]
    const int in_c, const int qout_c, const int out_c, const int group_size) {
  // Note: 16 * sizeof(float) = 64 bytes to fill a __m512
  TORCH_CHECK(out_c % 16 == 0);
  using scalar_vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;

#pragma omp parallel for
  for (int i = 0; i < in_c; ++i) {
    const int group_idx = i / group_size;
    const int *packed_weight_base = packed_weight + qout_c * i;
    const int *zeros_base = zeros + qout_c * group_idx;
    const scalar_t *scale_base = scale + out_c * group_idx;
    scalar_t *output_base = weight + out_c * i;
    const __m512 LUT =
        _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
                      7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    const __m512i OFFSET = _mm512_set_epi32(28, 12, 24, 8, 20, 4, 16, 0, 28, 12,
                                            24, 8, 20, 4, 16, 0);
    for (int j = 0; j < qout_c; j += 2) {
      vec_op::FP32Vec16 weight_vec =
          cast_packed_s4x16_to_fp32x16(packed_weight_base + j, LUT, OFFSET);
      vec_op::FP32Vec16 zero_vec =
          cast_packed_s4x16_to_fp32x16(zeros_base + j, LUT, OFFSET);

      scalar_vec_t scale_vec(scale_base + j * 8);
      vec_op::FP32Vec16 f32_scale_vec(scale_vec);
      vec_op::FP32Vec16 f32_dequantized_weight =
          (weight_vec - zero_vec) * f32_scale_vec;
      scalar_vec_t dequantize_weight(f32_dequantized_weight);
      dequantize_weight.save(output_base + j * 8);
    }
  }
}

// A block : {BLOCK_M, BLOCK_K}, ld=lda
// B block : {BLOCK_K, BLOCK_N / 8}, packed, ld=ldb
// zeros block : {BLOCK_K / G, BLOCK_N / 8}, packed, ld=ldb
// scales block: {BLOCK_K / G, BLOCK_N}, ld=ldb * 8
// C block : {BLOCK_M, BLOCK_N}, ld=ldb * 8
//
template <int BLOCK_M, int BLOCK_N, int G, typename scalar_t>
inline void int4gemm_kernel(const scalar_t *__restrict__ A,
                            const int *__restrict__ B,
                            const int *__restrict__ zeros,
                            const scalar_t *__restrict__ scales,
                            scalar_t *__restrict__ C, const int lda,
                            const int ldb, const int ldc, const int BLOCK_K) {
  using dot_vec_t = typename KernelVecType<scalar_t>::dot_vec_t;
  using scalar_vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int DOT_VEC_ELEM_NUM = dot_vec_t::get_elem_num();
  constexpr int SCALAR_VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  static_assert(DOT_VEC_ELEM_NUM == SCALAR_VEC_ELEM_NUM);

  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N / DOT_VEC_ELEM_NUM;

  //   TODO: Prefetch
  //   const int PREFETCH_SIZE_K = 16 * 4;
  //   const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;

  dot_vec_t vc[ROWS][COLS]; // Accumulator for C
  dot_vec_t vzero[COLS];    // zero cache vectors
  dot_vec_t vscale[COLS];   // scale cache vector

  // Lookup table and shuffle offset to de-quantize int4 values to dot type.
  const __m512 LUT =
      _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                    6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
  const __m512i OFFSET = _mm512_set_epi32(28, 12, 24, 8, 20, 4, 16, 0, 28, 12,
                                          24, 8, 20, 4, 16, 0);

  // Loop for BLOCK_K dim
  for (int i = 0; i < BLOCK_K; i += G) {
    // Load and dequant zeros, scales for each group
    const int group_idx = i / G;
    vec_op::unroll_loop<int, COLS>([&](int col_idx) {
      scalar_vec_t scale_v(scales + group_idx * ldb * 8 +
                           col_idx * SCALAR_VEC_ELEM_NUM);
      vscale[col_idx] = dot_vec_t(scale_v);
      vzero[col_idx] = cast_packed_s4x16_to_fp32x16(
          zeros + group_idx * ldb + col_idx * DOT_VEC_ELEM_NUM / 8, LUT,
          OFFSET);
    });

    for (int j = 0; j < G; ++j) {
      const int k_idx = i + j;
      dot_vec_t vb[COLS];
      // Load and dequantize weight
      vec_op::unroll_loop<int, COLS>([&](int col_idx) {
        dot_vec_t v = cast_packed_s4x16_to_fp32x16(
            B + k_idx * ldb + col_idx * DOT_VEC_ELEM_NUM / 8, LUT, OFFSET);
        vb[col_idx] = (v - vzero[col_idx]) * vscale[col_idx];
      });

      // Load A vector and compute with vb
      vec_op::unroll_loop<int, ROWS>([&](int row_idx) {
        scalar_t value = *(A + row_idx * lda + k_idx);
        scalar_vec_t vec = scalar_vec_t(value);
        dot_vec_t va = dot_vec_t(vec);
        vec_op::unroll_loop<int, COLS>([&](int col_idx) {
          vc[row_idx][col_idx] = vc[row_idx][col_idx] + va * vb[col_idx];
        });
      });
    }
  }

  // store to C
  vec_op::unroll_loop<int, ROWS>([&](int row_idx) {
    vec_op::unroll_loop<int, COLS>([&](int col_idx) {
      scalar_vec_t v(vc[row_idx][col_idx]);
      v.save(C + row_idx * ldb * 8 + col_idx * SCALAR_VEC_ELEM_NUM);
    });
  });
}

#define LAUNCH_INT4GEMM_KERNEL(MB_SIZE, NB_SIZE)                               \
  int4gemm_kernel<MB_SIZE, NB_SIZE, SHARED_GROUP_SIZE>(                        \
      sub_input_feature, sub_packed_weight, sub_zeros, sub_scales,             \
      sub_output_feature, num_in_channels, num_qout_channels,                  \
      num_out_channels, num_in_channels);

#define LAUNCH_INT4GEMM_NB_SIZE(MB_SIZE)                                       \
  switch (n_size) {                                                            \
  case 16:                                                                     \
    LAUNCH_INT4GEMM_KERNEL(MB_SIZE, 16);                                       \
    break;                                                                     \
  case 32:                                                                     \
    LAUNCH_INT4GEMM_KERNEL(MB_SIZE, 32);                                       \
    break;                                                                     \
  case 48:                                                                     \
    LAUNCH_INT4GEMM_KERNEL(MB_SIZE, 48);                                       \
    break;                                                                     \
  case 64:                                                                     \
    LAUNCH_INT4GEMM_KERNEL(MB_SIZE, 64);                                       \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported N block size: ", n_size);                  \
    break;                                                                     \
  }

template <typename scalar_t>
void awq_gemm_impl(
    const scalar_t
        *__restrict__ input_feature, // [num_in_feats, num_in_channels]
    const int
        *__restrict__ packed_weight,     // [num_in_channels, num_qout_channels]
    const int *__restrict__ zeros,       // [num_in_channels / group_size,
                                         // num_qout_channels]
    const scalar_t *__restrict__ scales, // [num_in_channels, num_out_channels]
    scalar_t *__restrict__ output_feature, //  [num_in_feats, num_out_channels]
    const int num_in_feats, const int num_in_channels,
    const int num_qout_channels, const int num_out_channels,
    const int group_size) {

  constexpr int dM = 4; // Partition size on num_in_feats dim
  constexpr int ACC_NUM = 16;
  constexpr int SHARED_GROUP_SIZE =
      128; // zeros and scales shared size on num_in_channels dim

  // TBD: Currently we hardcoded the group size as 128 to share the zeros and
  // scales among multiple in_channels.
  TORCH_CHECK(group_size % 128 == 0);

  using dot_vec_t = typename KernelVecType<scalar_t>::dot_vec_t;
  constexpr int DOT_VEC_ELEM_NUM = dot_vec_t::get_elem_num();
  static_assert(DOT_VEC_ELEM_NUM % 8 == 0);
  static_assert(ACC_NUM % dM == 0);
  static_assert(dM % 2 == 0);
  constexpr int dN =
      ACC_NUM / dM * DOT_VEC_ELEM_NUM; // Partition size on num_out_channels
  static_assert(dN % 8 == 0);
  TORCH_CHECK(num_out_channels % dN == 0);
  constexpr int DOT_VEC_MEM_LOAD_WIDTH =
      DOT_VEC_ELEM_NUM / 2 * ACC_NUM /
      dM; // mem load bytes from the weight per N partition
  constexpr int N_ALIGN_PARTITION_NUM = std::max(
      1,
      64 / DOT_VEC_MEM_LOAD_WIDTH); // Distribute N_ALIGN_PARTITION_NUM
                                    // partitions on num_out_channels dim to
                                    // same threads for cacheline alignment
  const int M =
      (num_in_feats + dM - 1) / dM;    // Partition num on num_in_feats dim
  const int N = num_out_channels / dN; // Partition num on num_out_feats dim
  const int MN = M * N;

//   std::printf("M: %d, N: %d\n", M, N);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < MN; i += N_ALIGN_PARTITION_NUM) {
    for (int j = 0; j < N_ALIGN_PARTITION_NUM; ++j) {
      const int m_idx = i / N;
      const int n_idx = i % N + j;
      const int m_size = std::min(dM, num_in_feats - m_idx * dM);
      const int n_size = std::min(dN, num_out_channels - n_idx * dN);

      const scalar_t *__restrict__ sub_input_feature =
          input_feature + m_idx * dM * num_in_channels;
      const int *__restrict__ sub_packed_weight =
          packed_weight + n_idx * dN / 8;
      const int *__restrict__ sub_zeros = zeros + n_idx * dN / 8;
      const scalar_t *__restrict__ sub_scales = scales + n_idx * dN;
      scalar_t *__restrict__ sub_output_feature =
          output_feature + m_idx * dM * num_out_channels + n_idx * dN;

      switch (m_size) {
      case 1:
        LAUNCH_INT4GEMM_NB_SIZE(1);
        break;
      case 2:
        LAUNCH_INT4GEMM_NB_SIZE(2);
        break;
      case 3:
        LAUNCH_INT4GEMM_NB_SIZE(3);
        break;
      case 4:
        LAUNCH_INT4GEMM_NB_SIZE(4);
        break;
      default:
        TORCH_CHECK(false, "Unsupported M block size: ", m_size);
      }
    }
  }
}
}; // namespace

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int split_k_iters, int thx,
                             int thy) {
  TORCH_CHECK_EQ(thx, 0);
  TORCH_CHECK_EQ(thy, 0);

  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  VLLM_DISPATCH_FLOATING_TYPES(
      _scaling_factors.scalar_type(), "awq_dequantize_impl", [&] {
        CPU_KERNEL_GUARD_IN(awq_dequantize_impl)
        awq_dequantize_impl(
            _kernel.data_ptr<int>(), _scaling_factors.data_ptr<scalar_t>(),
            _zeros.data_ptr<int>(), _de_kernel.data_ptr<scalar_t>(), in_c,
            qout_c, out_c, G);
        CPU_KERNEL_GUARD_OUT(awq_dequantize_impl)
      });

  return _de_kernel;
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int split_k_iters) {
  TORCH_CHECK_EQ(split_k_iters, 8);

  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  int num_out_channels = _kernel.size(1) * 8;
  int group_size = num_in_channels / _scaling_factors.size(0);

  TORCH_CHECK(group_size == 128);
  constexpr int GROUP_SIZE = 128;

  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
  at::Tensor _out_feats =
      torch::empty({num_in_feats, num_out_channels}, options);

  VLLM_DISPATCH_FLOATING_TYPES(_scaling_factors.scalar_type(), "awq_gemm", [&] {
    CPU_KERNEL_GUARD_IN(awq_gemm_impl)
    awq_gemm_impl(_in_feats.data_ptr<scalar_t>(), _kernel.data_ptr<int>(),
                  _zeros.data_ptr<int>(), _scaling_factors.data_ptr<scalar_t>(),
                  _out_feats.data_ptr<scalar_t>(), num_in_feats,
                  num_in_channels, num_out_channels / 8, num_out_channels,
                  group_size);
    CPU_KERNEL_GUARD_OUT(awq_gemm_impl)
  });
  return _out_feats;
}