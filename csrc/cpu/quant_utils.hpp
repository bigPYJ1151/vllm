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

#define DEFINE_PREPACK \
        template<typename WT, typename ST, typename ZT> \
        void prepack( \
            WT* __restrict__ weight_ptr, \
            ST* __restrict__ scale_ptr, \
            ZT* __restrict__ zp_ptr, \
            WT* __restrict__ packed_weight_ptr, \
            ST* __restrict__ packed_scale_ptr, \
            ZT* __restrict__ packed_zp_ptr, \
            const int32_t input_size \
        ) 

template<QuantMethod quant_method, cpu_utils::ISA isa>
class WeightProcessor {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 1;

DEFINE_PREPACK {
        TORCH_CHECK(false, "Unreachable"); 
    }
};

template<cpu_utils::ISA isa>
class WeightProcessor<QuantMethod::NONE, isa> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;
    static constexpr int32_t WEIGHT_PACK_FACTOR = 1;

    DEFINE_PREPACK {}
};

template<>
class WeightProcessor<QuantMethod::MXFP4, cpu_utils::ISA::VEC> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;

    DEFINE_PREPACK {
        TORCH_CHECK(false, "Unreachable"); 
    }
};

template<>
class WeightProcessor<QuantMethod::MXFP4, cpu_utils::ISA::AMX> {
public:
    static constexpr int32_t OUTPUT_BLOCK_SIZE = 16;

    DEFINE_PREPACK{
        static_assert(std::is_same_v<WT, int32_t>);
        static_assert(std::is_same_v<ST, uint8_t>);

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
