#ifndef QUANT_UTILS_HPP
#define QUANT_UTILS_HPP

#include "cpu/utils.hpp"

namespace cpu_quant_utils {
enum class QuantMethod { NONE, MXFP4 };

inline QuantMethod get_quantmethod(const std::string& method) {
  if (method == "none") {
    return QuantMethod::NONE;
  } else if (method == "mxfp4") {
    return QuantMethod::MXFP4;
  } else {
    TORCH_CHECK(false, "Invalid quant method type: " + method);
  }
}

#define DEFINE_WP_METHODS \
    private: \
        weight_t* __restrict__ curr_input_weight_ptr_; \
        scalar_t* __restrict__ curr_output_weight_ptr_; \
        scalar_t* __restrict__ output_weight_ptr_; \
        scale_t* __restrict__ curr_scale_ptr_; \
        zero_point_t* __restrict__ curr_zero_point_ptr_; \
    \
    public: \
        FORCE_INLINE scalar_t* get_processed_weight_ptr() const {return output_weight_ptr_;} \
\
        FORCE_INLINE WeightProcessor( \
            weight_t* __restrict__ weight_ptr, \
            scalar_t* __restrict__ output_weight_ptr, \
            scale_t* __restrict__ scale_ptr, \
            zero_point_t* __restrict__ zp_ptr, \
            const int32_t input_size, \
            const int32_t output_size, \
            const int32_t expert_idx, \
            const int32_t output_idx \
        ) \
        \

#define DEFINE_WP_PREPACK \
        void prepack( \
            weight_t* __restrict__ weight_ptr, \
            scale_t* __restrict__ scale_ptr, \
            zero_point_t* __restrict__ zp_ptr, \
            weight_t* __restrict__ packed_weight_ptr, \
            scale_t* __restrict__ packed_scale_ptr, \
            zero_point_t* __restrict__ packed_zp_ptr, \
            const int32_t input_size \
        )

template<QuantMethod quant_method, cpu_utils::ISA isa, typename scalar_t>
class WeightProcessor {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 1;
    using weight_t = scalar_t;
    using scale_t = void;
    using zero_point_t = void;

DEFINE_WP_METHODS {
        TORCH_CHECK(false, "Not implemented"); 
    }
};

template<cpu_utils::ISA isa, typename scalar_t>
class WeightProcessor<QuantMethod::NONE, isa, scalar_t> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 1;
    using weight_t = scalar_t;
    using scale_t = void;
    using zero_point_t = void;

    DEFINE_WP_METHODS: 
    curr_input_weight_ptr_(weight_ptr + (expert_idx * input_size * output_size + output_idx * input_size) / WEIGHT_PACK_FACTOR),
    curr_output_weight_ptr_(curr_input_weight_ptr_),
    output_weight_ptr_(curr_input_weight_ptr_),
    curr_scale_ptr_(nullptr),
    curr_zero_point_ptr_(nullptr)
    {}

    DEFINE_WP_PREPACK {}

};

namespace  {
alignas(64) static const uint16_t MXFP4_TO_BF16_LUT[32] = {
        0x0000,
        0x3f00,
        0x3f80,
        0x3fc0,
        0x4000,
        0x4040,
        0x4080,
        0x40c0,
        0x0000, // convert -0 to +0
        0xbf00,
        0xbf80,
        0xbfc0,
        0xc000,
        0xc040,
        0xc080,
        0xc0c0,
// padding
        0x0000, 
        0x3f00,
        0x3f80,
        0x3fc0,
        0x4000,
        0x4040,
        0x4080,
        0x40c0,
        0x0000,
        0xbf00,
        0xbf80,
        0xbfc0,
        0xc000,
        0xc040,
        0xc080,
        0xc0c0
};
}

template<>
class WeightProcessor<QuantMethod::MXFP4, cpu_utils::ISA::VEC, c10::BFloat16> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 4;
    using weight_t = int32_t;
    using scale_t = uint8_t;
    using zero_point_t = void;

    DEFINE_WP_PREPACK {
        TORCH_CHECK(false, "Unreachable"); 
    }
};

template<>
class WeightProcessor<QuantMethod::MXFP4, cpu_utils::ISA::AMX, c10::BFloat16> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 4;
    static constexpr int32_t SCALE_BLOCK_SIZE = 32;
    using scalar_t = c10::BFloat16;
    using weight_t = int32_t;
    using scale_t = uint8_t;
    using zero_point_t = void;

    DEFINE_WP_METHODS:
    curr_input_weight_ptr_(weight_ptr + (expert_idx * input_size * output_size + output_idx * input_size) / WEIGHT_PACK_FACTOR),
    curr_output_weight_ptr_(output_weight_ptr),
    output_weight_ptr_(curr_output_weight_ptr_),
    curr_scale_ptr_(scale_ptr + (expert_idx * input_size * output_size + output_idx * input_size) / SCALE_BLOCK_SIZE),
    curr_zero_point_ptr_(nullptr)
    {}

    FORCE_INLINE void dequant(const int32_t tile_size_n, const int32_t tile_size_k) {
        const int32_t n_iterations = tile_size_n / 16;
        const int32_t k_iterations = tile_size_k / 32;
        scalar_t* __restrict__ curr_output_ptr = curr_output_weight_ptr_;
        weight_t* __restrict__ curr_input_ptr = curr_input_weight_ptr_;
        scale_t* __restrict__ curr_scale_ptr = curr_scale_ptr_;

        vec_op::BF16Vec32 mxfp4_to_bf16_lut(MXFP4_TO_BF16_LUT);
        for (int32_t n = 0; n < n_iterations; ++n) {
            for (int32_t k = 0; k < k_iterations; ++k) {
                // in each iteration, dequant [16, 32] elements
                // load scales for 16 x 32 blocks, extand to 16 bits, and duplicate for VNNI format
                __m128i scales = _mm_loadu_epi8(curr_scale_ptr);
                __m512i scales_512 = _mm512_cvtepu8_epi32(scales);
                scales_512 = _mm512_or_epi32(scales_512, _mm512_slli_epi32(scales_512, 16));
                // shift scales to BF16 exponent bits 
                vec_op::BF16Vec32 scales_vec(_mm512_slli_epi32(scales_512, 7));

                for (int32_t i = 0; i < 4; ++i) {
                    vec_op::BF16Vec32 packed_mxfp4_vec(curr_input_ptr);
                    for (int32_t j = 0; j < 4; ++j) {
                        vec_op::BF16Vec32 value_vec = vec_op::convert_mxfp4_to_bf16(packed_mxfp4_vec >> (j * 4), mxfp4_to_bf16_lut, scales_vec);
                        value_vec.save(curr_output_ptr);
                        // update
                        curr_output_ptr += 32;
                    }
                    // update
                    curr_input_ptr += 16;
                }

                // update
                curr_scale_ptr += 16;
            }
        }
    }

    DEFINE_WP_PREPACK{
        static_assert(std::is_same_v<weight_t, int32_t>);
        static_assert(std::is_same_v<scale_t, uint8_t>);

        // pack scale, just transpose [OUTPUT_BLOCK_SIZE, input_size // 32]  
        {
            const int32_t scale_input_size = input_size / 32; 
            for (int32_t i = 0; i < scale_input_size; ++i) {
                for (int32_t j = 0; j < OUTPUT_BLOCK_SIZE; ++j) {
                    packed_scale_ptr[j] = *(scale_ptr + scale_input_size * j);
                }
                scale_ptr += 1;
                packed_scale_ptr += OUTPUT_BLOCK_SIZE;
            }
        }

        // pack weight, 
        // - transpose int32 blocks
        // - within each block, shuffle [x0, x1, x2, x3, x4, x5, x6, x7] to [x0, x2, x4, x6, x1, x3, x5, x7]
        {
            const int32_t input_size_32b = input_size / 8;
            constexpr int32_t mask_4b = 0xF;
            for (int32_t i = 0; i < input_size_32b; ++i) {
                for (int32_t j = 0; j < OUTPUT_BLOCK_SIZE; ++j) {
                    int32_t block_32b = *(weight_ptr + input_size_32b * j);
                    int32_t shuffled_block_32b = 0;
                    shuffled_block_32b |= (block_32b & mask_4b); // x0
                    shuffled_block_32b |= (((block_32b >> 8) & mask_4b) << 4) ; // x2
                    shuffled_block_32b |= (((block_32b >> 16) & mask_4b) << 8) ; // x4
                    shuffled_block_32b |= (((block_32b >> 24) & mask_4b) << 12) ; // x6
                    shuffled_block_32b |= (((block_32b >> 4) & mask_4b) << 16) ; // x1
                    shuffled_block_32b |= (((block_32b >> 12) & mask_4b) << 20) ; // x3
                    shuffled_block_32b |= (((block_32b >> 20) & mask_4b) << 24) ; // x5
                    shuffled_block_32b |= (((block_32b >> 28) & mask_4b) << 28) ; // x7
                    packed_weight_ptr[j] = shuffled_block_32b;
                }
                weight_ptr += 1;
                packed_weight_ptr += OUTPUT_BLOCK_SIZE;
            }
        }
    }
};
}
#endif // QUANT_UTILS_HPP
