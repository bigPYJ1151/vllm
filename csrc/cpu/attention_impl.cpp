#include "cpu_types.hpp"

namespace {
template <typename T>
inline std::pair<T, T> reduceSoftmax(T *data, const int size,
                                     const int capacity) {
  T max = data[0];
  for (int i = 1; i < size; ++i) {
    max = max >= data[i] ? max : data[i];
  }

  T sum = 0;
  for (int i = 0; i < size; ++i) {
    data[i] = std::exp(data[i] - max);
    sum += data[i];
  }

  int i = 0;
  for (; i < size; ++i) {
    data[i] /= sum;
  }

  for (; i < capacity; ++i) {
    data[i] = 0;
  }

  return {max, sum};
}

template <typename T>
inline void reducePartitonSoftmax(const T *max_data, T *sum_data,
                                  const int size) {
  T max = max_data[0];
  for (int i = 1; i < size; ++i) {
    max = max >= max_data[i] ? max : max_data[i];
  }

  T rescaled_sum = 0;
  for (int i = 0; i < size; ++i) {
    T rescale_factor = std::exp(max_data[i] - max);
    rescaled_sum += rescale_factor * sum_data[i];
    sum_data[i] *= rescale_factor;
  }
  for (int i = 0; i < size; ++i) {
    sum_data[i] /= rescaled_sum + 1e-8;
  }
}

// Note: not a efficient implementation for fp32
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int x>
void reduceQKBlock(const scalar_t *__restrict__ q,
                   const scalar_t *__restrict__ k_block,
                   float *__restrict__ logits, float scale, bool is_tail) {
  for (int q_offset = 0; q_offset < HEAD_SIZE; q_offset += x, q += x) {
    for (int token_idx = 0; token_idx < BLOCK_SIZE; ++token_idx, k_block += x) {
      for (int i = 0; i < x; ++i) {
        logits[token_idx] += q[i] * k_block[i] * scale;
      }
    }
  }
}

template <int HEAD_SIZE, int BLOCK_SIZE, int x>
void reduceQKBlock(const c10::BFloat16 *__restrict__ q,
                   const c10::BFloat16 *__restrict__ k_block,
                   float *__restrict__ logits, float scale, bool is_tail) {
  static_assert(vec_op::BF16Vec32::get_elem_num() % x == 0);
  constexpr int TOKEN_PER_GROUP = vec_op::BF16Vec32::get_elem_num() / x;
  static_assert(BLOCK_SIZE % TOKEN_PER_GROUP == 0);
  constexpr int TOKEN_GROUPS = BLOCK_SIZE / TOKEN_PER_GROUP;

  vec_op::FP32Vec16 group_accums[TOKEN_GROUPS];

  for (int q_offset = 0; q_offset < HEAD_SIZE;
       q_offset += x, k_block += x * BLOCK_SIZE) {
    vec_op::BF16Vec8 q_vec(q + q_offset);
    vec_op::BF16Vec32 q_group_vec(q_vec);

    vec_op::unroll_loop<int, TOKEN_GROUPS>(
        [k_block, &q_group_vec, &group_accums](int token_group_idx) {
          vec_op::BF16Vec32 k_group_vec(k_block +
                                        token_group_idx * x * TOKEN_PER_GROUP);

          group_accums[token_group_idx] = vec_op::fma(
              q_group_vec, k_group_vec, group_accums[token_group_idx]);
          vec_op::prefetch(k_block + x * BLOCK_SIZE +
                           token_group_idx * x * TOKEN_PER_GROUP);
        });
  }

  vec_op::unroll_loop<int, TOKEN_GROUPS>([&group_accums, logits,
                                          scale](int token_group_idx) {
    vec_op::unroll_loop<int, TOKEN_PER_GROUP>(
        [&group_accums, logits, scale, token_group_idx](int token_idx) {
          float dot_v =
              group_accums[token_group_idx]
                  .template reduce_sub_sum<vec_op::FP32Vec16::get_elem_num() /
                                           TOKEN_PER_GROUP>(token_idx);
          logits[token_group_idx * TOKEN_PER_GROUP + token_idx] = dot_v * scale;
        });
  });
}

// Note: not a efficient implementation for fp32
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
          int HEAD_PARTITION_SIZE, typename acc_t>
void reduceValueBlock(const float *prob, const scalar_t *v_block, acc_t &&acc) {
  for (int i = 0; i < 16; ++i, v_block += BLOCK_SIZE) {
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      acc[i] += prob[j] * v_block[j];
    }
  }
}

template <int HEAD_SIZE, int BLOCK_SIZE, int HEAD_PARTITION_SIZE,
          typename acc_t>
void reduceValueBlock(const float *prob, const c10::BFloat16 *v_block,
                      acc_t &&acc) {
  vec_op::FP32Vec16 prob_vec(prob);

  vec_op::unroll_loop<int, HEAD_PARTITION_SIZE>([&](int head_elem_idx) {
    vec_op::BF16Vec16 v_vec(v_block + BLOCK_SIZE * head_elem_idx);
    vec_op::FP32Vec16 fp32_v_vec(v_vec.reg);
    acc[head_elem_idx] = acc[head_elem_idx] + prob_vec * fp32_v_vec;
  });
}
}; // namespace

// Paged attention v1
namespace {
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl {
  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const float *__restrict__ alibi_slopes, // [num_heads]
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    TORCH_CHECK(HEAD_SIZE % 16 == 0);
    TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    size_t logits_bytes = num_heads * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [head_num, max_context_len_padded]

    std::memset(out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int context_len = context_lens[seq_idx];
      const int *seq_block_table =
          block_tables + max_num_blocks_per_seq * seq_idx;
      const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
      std::memset(logits, 0, logits_bytes);

      // Compute attention logits
#pragma omp parallel for collapse(2)
      for (int block_idx = 0; block_idx < block_num; ++block_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t *__restrict__ q_vec_ptr =
              q + seq_idx * q_stride + head_idx * HEAD_SIZE;
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float *__restrict__ head_block_logits =
              logits + head_idx * max_context_len_padded +
              block_idx * BLOCK_SIZE;

          reduceQKBlock<scalar_t, HEAD_SIZE, BLOCK_SIZE, x>(
              q_vec_ptr, k_block_cache_ptr, head_block_logits, scale, false);
        }
      }

      // Compute softmax
#pragma omp parallel for
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        float *head_logit_ptr = logits + head_idx * max_context_len_padded;
        reduceSoftmax(head_logit_ptr, context_len, block_num * BLOCK_SIZE);
      }

      // Compute value
      constexpr int head_partition_num = HEAD_SIZE / 16;
#pragma omp parallel for collapse(2)
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t kv_head_idx = head_idx / num_queries_per_kv;
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float *__restrict__ prob_vec_ptr =
                logits + head_idx * max_context_len_padded +
                block_idx * BLOCK_SIZE;
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride + BLOCK_SIZE * head_part_idx * 16;
            scalar_t *__restrict__ out_ptr =
                out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
                head_part_idx * 16;
            reduceValueBlock<scalar_t, HEAD_SIZE, BLOCK_SIZE, 16>(
                prob_vec_ptr, v_block_cache_ptr, out_ptr);
          }
        }
      }
    }
    std::free(logits);
  }
};

template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl<c10::BFloat16, HEAD_SIZE, BLOCK_SIZE> {
  using scalar_t = c10::BFloat16;

  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const float *__restrict__ alibi_slopes, // [num_heads]
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    using scalar_vec_t = vec_op::vec_t<scalar_t>;
    constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

    static_assert(x == VEC_ELEM_NUM);
    static_assert(BLOCK_SIZE == 16);
    static_assert(BLOCK_SIZE % VEC_ELEM_NUM == 0);

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    const int parallel_work_item_num = omp_get_max_threads();

    size_t logits_bytes =
        parallel_work_item_num * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [parallel_work_item_num, max_context_len_padded]

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int context_len = context_lens[seq_idx];
        const int *seq_block_table =
            block_tables + max_num_blocks_per_seq * seq_idx;
        const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int64_t kv_head_idx = head_idx / num_queries_per_kv;
        const scalar_t *__restrict__ q_vec_ptr =
            q + seq_idx * q_stride + head_idx * HEAD_SIZE;
        float *__restrict__ thread_block_logits =
            logits + omp_get_thread_num() * max_context_len_padded;

        // Compute logits
        for (int block_idx = 0; block_idx < block_num; ++block_idx) {
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float *__restrict__ head_block_logits =
              thread_block_logits + block_idx * BLOCK_SIZE;

          reduceQKBlock<HEAD_SIZE, BLOCK_SIZE, x>(
              q_vec_ptr, k_block_cache_ptr, head_block_logits, scale, false);
        }

        // Compute softmax
        reduceSoftmax(thread_block_logits, context_len, block_num * BLOCK_SIZE);

        // Compute value
        constexpr int head_elem_num_per_partition = 16;
        constexpr int head_partition_num =
            HEAD_SIZE / head_elem_num_per_partition;
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          vec_op::FP32Vec16 accums[head_elem_num_per_partition];
          scalar_t *__restrict__ out_ptr =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              head_part_idx * head_elem_num_per_partition;
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float *__restrict__ prob_vec_ptr =
                thread_block_logits + block_idx * BLOCK_SIZE;
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride +
                BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
            reduceValueBlock<HEAD_SIZE, BLOCK_SIZE,
                             head_elem_num_per_partition>(
                prob_vec_ptr, v_block_cache_ptr, accums);
          }

          vec_op::unroll_loop<int, head_elem_num_per_partition>(
              [&](int head_elem_idx) {
                float value = accums[head_elem_idx].reduce_sum();
                vec_op::storeFP32ToT(value, out_ptr + head_elem_idx);
              });
        }
      }
    }
    std::free(logits);
  }
};

#define LAUNCH_V1_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                   \
  paged_attention_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call(                     \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale, \
      block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,              \
      alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads);

template <typename T, int BLOCK_SIZE>
void paged_attention_v1_impl_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  switch (head_size) {
  case 64:
    LAUNCH_V1_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_V1_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_V1_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_V1_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_V1_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  case 256:
    LAUNCH_V1_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_V1_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                 \
  paged_attention_v1_impl_launcher<T, BLOCK_SIZE>(                             \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,   \
      context_lens, max_context_len, alibi_slopes);

#define CALL_V1_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                  \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_V1_KERNEL_LAUNCHER(T, 16);                                            \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }
} // namespace

void paged_attention_v1_cpu(torch::Tensor &out, torch::Tensor &query,
                            torch::Tensor &key_cache,
                            torch::Tensor &value_cache, int num_kv_heads,
                            float scale, torch::Tensor &block_tables,
                            torch::Tensor &context_lens, int block_size,
                            int max_context_len,
                            const c10::optional<torch::Tensor> &alibi_slopes) {
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "paged_attention_v1_impl",
                               [&] {
                                 CPU_KERNEL_GUARD_IN(paged_attention_v1_impl)
                                 CALL_V1_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
                                 CPU_KERNEL_GUARD_OUT(paged_attention_v1_impl)
                               });
}

// Paged attention v2
namespace {
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int PARTITION_SIZE>
struct paged_attention_v2_impl {
  static void call(
      scalar_t *__restrict__ out,   // [num_seqs, num_heads, head_size]
      float *__restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
      float
          *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
      scalar_t *__restrict__ tmp_out,       // [num_seqs, num_heads,
                                            // max_num_partitions, head_size]
      const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
      const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                            // head_size/x, block_size, x]
      const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                            // head_size, block_size]
      const int num_kv_heads, const float scale,
      const int
          *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
      const int *__restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float *__restrict__ alibi_slopes, // [num_heads]
      const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int num_seqs, const int num_heads, const int max_num_partitions) {
    paged_attention_v1_impl<scalar_t, HEAD_SIZE, BLOCK_SIZE>::call(
        out, q, k_cache, v_cache, num_kv_heads, scale, block_tables,
        context_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
        kv_block_stride, kv_head_stride, num_seqs, num_heads);
  }
};

template <int HEAD_SIZE, int BLOCK_SIZE, int PARTITION_SIZE>
struct paged_attention_v2_impl<c10::BFloat16, HEAD_SIZE, BLOCK_SIZE,
                               PARTITION_SIZE> {
  using scalar_t = c10::BFloat16;

  static void call(
      scalar_t *__restrict__ out,   // [num_seqs, num_heads, head_size]
      float *__restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
      float
          *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
      scalar_t *__restrict__ tmp_out,       // [num_seqs, num_heads,
                                            // max_num_partitions, head_size]
      const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
      const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                            // head_size/x, block_size, x]
      const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                            // head_size, block_size]
      const int num_kv_heads, const float scale,
      const int
          *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
      const int *__restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float *__restrict__ alibi_slopes, // [num_heads]
      const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int num_seqs, const int num_heads, const int max_num_partitions) {
    TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    using scalar_vec_t = vec_op::vec_t<scalar_t>;
    constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
    constexpr int BLOCK_NUM_PER_PARTITION = PARTITION_SIZE / BLOCK_SIZE;

    static_assert(x == VEC_ELEM_NUM);
    static_assert(BLOCK_SIZE == 16);
    static_assert(BLOCK_SIZE % VEC_ELEM_NUM == 0);
    static_assert(PARTITION_SIZE * sizeof(float) % 64 == 0);
    static_assert(PARTITION_SIZE % BLOCK_SIZE == 0);

#pragma omp parallel for collapse(3) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int partition_idx = 0; partition_idx < max_num_partitions;
           ++partition_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
          const int context_len = context_lens[seq_idx];
          const int start_token_idx = partition_idx * PARTITION_SIZE;

          if (start_token_idx >= context_len)
            continue;

          const int partition_num =
              (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
          const bool no_reduce = (partition_num == 1);
          const int context_token_num =
              (std::min(context_len, start_token_idx + PARTITION_SIZE) -
               start_token_idx);
          const int block_num =
              (context_token_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const int *seq_block_table = block_tables +
                                       max_num_blocks_per_seq * seq_idx +
                                       start_token_idx / BLOCK_SIZE;
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          const scalar_t *__restrict__ q_vec_ptr =
              q + seq_idx * q_stride + head_idx * HEAD_SIZE;

          float logits[PARTITION_SIZE] __attribute__((aligned(64)));

          // Compute logits
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const scalar_t *__restrict__ k_block_cache_ptr =
                k_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride;
            float *__restrict__ head_block_logits =
                logits + block_idx * BLOCK_SIZE;

            reduceQKBlock<HEAD_SIZE, BLOCK_SIZE, x>(
                q_vec_ptr, k_block_cache_ptr, head_block_logits, scale, false);
          }

          auto &&[max_logit, exp_sum] =
              reduceSoftmax(logits, context_token_num, block_num * BLOCK_SIZE);

          scalar_t *__restrict__ output_buffer = nullptr;
          if (!no_reduce) {
            auto idx = seq_idx * num_heads * max_num_partitions +
                       head_idx * max_num_partitions + partition_idx;
            max_logits[idx] = max_logit;
            exp_sums[idx] = exp_sum;
            output_buffer =
                tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                head_idx * max_num_partitions * HEAD_SIZE +
                partition_idx * HEAD_SIZE;
          } else {
            output_buffer =
                out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
          }

          // Compute value
          constexpr int head_elem_num_per_partition = 16;
          constexpr int head_partition_num =
              HEAD_SIZE / head_elem_num_per_partition;
          for (int head_part_idx = 0; head_part_idx < head_partition_num;
               ++head_part_idx) {
            vec_op::FP32Vec16 accums[head_elem_num_per_partition];
            scalar_t *__restrict__ out_ptr =
                output_buffer + head_part_idx * head_elem_num_per_partition;
            for (int block_idx = 0; block_idx < block_num; ++block_idx) {
              const int64_t physical_block_idx = seq_block_table[block_idx];
              const float *__restrict__ prob_vec_ptr =
                  logits + block_idx * BLOCK_SIZE;
              const scalar_t *__restrict__ v_block_cache_ptr =
                  v_cache + physical_block_idx * kv_block_stride +
                  kv_head_idx * kv_head_stride +
                  BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
              reduceValueBlock<HEAD_SIZE, BLOCK_SIZE,
                               head_elem_num_per_partition>(
                  prob_vec_ptr, v_block_cache_ptr, accums);
            }

            vec_op::unroll_loop<int, head_elem_num_per_partition>(
                [&](int head_elem_idx) {
                  float value = accums[head_elem_idx].reduce_sum();
                  vec_op::storeFP32ToT(value, out_ptr + head_elem_idx);
                });
          }
        }
      }
    }

    // Rescale partition softmax and store the factors to exp_sums
#pragma omp parallel for collapse(2) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int context_len = context_lens[seq_idx];
        const int partition_num =
            (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

        if (partition_num == 1)
          continue;

        reducePartitonSoftmax(
            max_logits + seq_idx * num_heads * max_num_partitions +
                head_idx * max_num_partitions,
            exp_sums + seq_idx * num_heads * max_num_partitions +
                head_idx * max_num_partitions,
            partition_num);
      }
    }

    // Reduce values
    constexpr int head_elem_num_per_group =
        16; // Note: didn't align with the cacheline size, due to some HEAD_SIZE
            // didn't align with 64 bytes
    static_assert(HEAD_SIZE % head_elem_num_per_group == 0);
    constexpr int head_group_num = HEAD_SIZE / head_elem_num_per_group;
    const float *__restrict__ rescale_factors = exp_sums;
#pragma omp parallel for collapse(3) schedule(static, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (int group_idx = 0; group_idx < head_group_num; ++group_idx) {
          const int context_len = context_lens[seq_idx];
          const int partition_num =
              (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

          if (partition_num == 1)
            continue;

          const float *__restrict__ seq_head_rescale_factors =
              rescale_factors + seq_idx * num_heads * max_num_partitions +
              head_idx * max_num_partitions;
          const scalar_t *__restrict__ seq_head_tmp_out =
              tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
              head_idx * max_num_partitions * HEAD_SIZE +
              group_idx * head_elem_num_per_group;
          scalar_t *__restrict__ seq_head_output =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              group_idx * head_elem_num_per_group;

          vec_op::FP32Vec16 acc;
          for (int i = 0; i < partition_num; ++i) {
            vec_op::FP32Vec16 rescale_factor(seq_head_rescale_factors[i]);
            vec_op::BF16Vec16 bf16_value(seq_head_tmp_out + i * HEAD_SIZE);
            vec_op::FP32Vec16 fp32_value(bf16_value.reg);
            acc = acc + fp32_value * rescale_factor;
          }
          vec_op::BF16Vec16 bf16_acc(acc.reg);
          bf16_acc.save(seq_head_output);
        }
      }
    }
  }
};

#define LAUNCH_V2_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                   \
  paged_attention_v2_impl<T, HEAD_SIZE, BLOCK_SIZE, PARTITION_SIZE>::call(     \
      out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr,           \
      key_cache_ptr, value_cache_ptr, num_kv_heads, scale, block_tables_ptr,   \
      context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
      kv_block_stride, kv_head_stride, num_seqs, num_heads,                    \
      max_num_partitions);

template <typename T, int BLOCK_SIZE, int PARTITION_SIZE = 512>
void paged_attention_v2_impl_launcher(
    torch::Tensor &out, torch::Tensor &exp_sums, torch::Tensor &max_logits,
    torch::Tensor &tmp_out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);
  int max_num_partitions = exp_sums.size(-1);

  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  float *exp_sums_ptr = reinterpret_cast<float *>(exp_sums.data_ptr());
  float *max_logits_ptr = reinterpret_cast<float *>(max_logits.data_ptr());
  T *tmp_out_ptr = reinterpret_cast<T *>(tmp_out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  switch (head_size) {
  case 64:
    LAUNCH_V2_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_V2_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_V2_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_V2_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_V2_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  case 256:
    LAUNCH_V2_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_V2_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                 \
  paged_attention_v2_impl_launcher<T, BLOCK_SIZE>(                             \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, block_size,             \
      max_context_len, alibi_slopes);

#define CALL_V2_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                  \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_V2_KERNEL_LAUNCHER(T, 16);                                            \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }
} // namespace

void paged_attention_v2_cpu(torch::Tensor &out, torch::Tensor &exp_sums,
                            torch::Tensor &max_logits, torch::Tensor &tmp_out,
                            torch::Tensor &query, torch::Tensor &key_cache,
                            torch::Tensor &value_cache, int num_kv_heads,
                            float scale, torch::Tensor &block_tables,
                            torch::Tensor &context_lens, int block_size,
                            int max_context_len,
                            const c10::optional<torch::Tensor> &alibi_slopes) {
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "paged_attention_v2_impl",
                               [&] {
                                 CPU_KERNEL_GUARD_IN(paged_attention_v2_impl)
                                 CALL_V2_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
                                 CPU_KERNEL_GUARD_OUT(paged_attention_v2_impl)
                               });
}