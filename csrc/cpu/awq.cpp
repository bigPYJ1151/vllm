#include "cpu_types.hpp"

namespace {
template <typename scalar_t> struct KernelVecType {
  using scalar_vec_t = void;
};

template <> struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
};

template <> struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
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