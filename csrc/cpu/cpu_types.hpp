
#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP

#include <immintrin.h>
#include <torch/extension.h>

namespace vec_op {

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

#ifdef __AVX512FP16__
struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128h reg;

  explicit FP16Vec8(_Float16 v) : reg(_mm_set1_ph(v)) {}

  explicit FP16Vec8(const void *ptr) : reg(_mm_loadu_ph(ptr)) {}

  explicit FP16Vec8(__m128h data) : reg(data) {}

  explicit FP16Vec8(__m256 v)
      : reg(_mm_castsi128_ph(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT))){};

  FP16Vec8 operator*(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_mul_ph(reg, b.reg));
  }

  FP16Vec8 operator+(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_add_ph(reg, b.reg));
  }

  FP16Vec8 operator-(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_sub_ph(reg, b.reg));
  }

  FP16Vec8 operator/(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_div_ph(reg, b.reg));
  }

  void save(void *ptr) const { _mm_storeu_ph(ptr, reg); }
};
#endif

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128bh reg;

  explicit BF16Vec8(const void *ptr)
      : reg(*reinterpret_cast<const __m128bh *>(ptr)) {}

  explicit BF16Vec8(__m128bh data) : reg(data) {}

  explicit BF16Vec8(__m256 v) : reg(_mm256_cvtneps_pbh(v)){};

  void save(void *ptr) const { *reinterpret_cast<__m128bh *>(ptr) = reg; }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    __m256 reg;
    float values[8];
  };

  __m256 reg;

  explicit FP32Vec8(float v) : reg(_mm256_set1_ps(v)) {}

  explicit FP32Vec8() : reg(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec8(const float *ptr) : reg(_mm256_loadu_ps(ptr)) {}

  explicit FP32Vec8(__m256 data) : reg(data) {}

#ifdef __AVX512FP16__
  explicit FP32Vec8(__m128h v) : reg(_mm256_cvtph_ps(_mm_castph_si128(v))) {}
#endif

  explicit FP32Vec8(__m128bh v) : reg(_mm256_cvtpbh_ps(v)) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    for (int i = 0; i < 8; ++i) {
      ans += ar.values[i];
    }
    return ans;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(expf(ar.values[7]), expf(ar.values[6]),
                                  expf(ar.values[5]), expf(ar.values[4]),
                                  expf(ar.values[3]), expf(ar.values[2]),
                                  expf(ar.values[1]), expf(ar.values[0])));
  }

  FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_mul_ps(reg, b.reg));
  }

  FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_add_ps(reg, b.reg));
  }

  FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_sub_ps(reg, b.reg));
  }

  FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_div_ps(reg, b.reg));
  }

  void save(float *ptr) const { _mm256_storeu_ps(ptr, reg); }
};

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

#ifdef __AVX512FP16__
template <> struct VecType<c10::Half> { using vec_type = FP16Vec8; };
#endif

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

template <typename T> void storeFP32ToT(float v, T *ptr) { *ptr = v; }

#ifdef __AVX512FP16__
template <> inline void storeFP32ToT<c10::Half>(float v, c10::Half *ptr) {
  *reinterpret_cast<_Float16 *>(ptr) = v;
}
#endif

template <>
inline void storeFP32ToT<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  *reinterpret_cast<__bfloat16 *>(ptr) = _mm_cvtness_sbh(v);
}

}; // namespace vec_op

#endif