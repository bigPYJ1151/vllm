#include <c10/util/Optional.h>
#include <cfloat>
#include <cmath>
#include <torch/extension.h>

namespace {

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
void single_query_cached_kv_attention_impl(
    scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int *__restrict__ head_mapping, // [num_heads]
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const int num_seqs, const int num_heads) {
  constexpr int x = 16 / sizeof(scalar_t);

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
        const int kv_head_idx = head_mapping[head_idx];
        const int physical_block_idx = seq_block_table[block_idx];
        const scalar_t *__restrict__ q_vec_ptr =
            q + seq_idx * q_stride + head_idx * HEAD_SIZE;
        const scalar_t *__restrict__ k_block_cache_ptr =
            k_cache + physical_block_idx * kv_block_stride +
            kv_head_idx * kv_head_stride;
        float *__restrict__ head_block_logits =
            logits + head_idx * max_context_len_padded + block_idx * BLOCK_SIZE;

        for (int q_offset = 0; q_offset < HEAD_SIZE;
             q_offset += x, q_vec_ptr += x) {
          for (int token_idx = 0; token_idx < BLOCK_SIZE;
               ++token_idx, k_block_cache_ptr += x) {
            for (int i = 0; i < x; ++i) {
              head_block_logits[token_idx] +=
                  q_vec_ptr[i] * k_block_cache_ptr[i] * scale;
            }
          }
        }
      }
    }

    // Compute softmax
#pragma omp parallel for
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      float *head_logit_ptr = logits + head_idx * max_context_len_padded;
      float max_logit = head_logit_ptr[0];
      for (int i = 1; i < context_len; ++i) {
        max_logit =
            max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
      }

      float sum = 0;
      for (int i = 0; i < context_len; ++i) {
        head_logit_ptr[i] = std::exp(head_logit_ptr[i] - max_logit);
        sum += head_logit_ptr[i];
      }

      for (int i = 0; i < context_len; ++i) {
        head_logit_ptr[i] /= sum;
      }

      int remaining_seq_upper = block_num * BLOCK_SIZE;
      for (int i = context_len; i < remaining_seq_upper; ++i) {
        head_logit_ptr[i] = 0;
      }
    }

    // Compute value
#pragma omp parallel for collapse(2)
    for (int block_idx = 0; block_idx < block_num; ++block_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int kv_head_idx = head_mapping[head_idx];
        const int physical_block_idx = seq_block_table[block_idx];
        const scalar_t *__restrict__ prob_vec_ptr =
            logits + head_idx * max_context_len_padded + block_idx * BLOCK_SIZE;
        const scalar_t *__restrict__ v_block_cache_ptr =
            v_cache + physical_block_idx * kv_block_stride +
            kv_head_idx * kv_head_stride;
        scalar_t *__restrict__ out_ptr =
            out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

        for (int i = 0; i < HEAD_SIZE; ++i, v_block_cache_ptr += BLOCK_SIZE) {
          for (int j = 0; j < BLOCK_SIZE; ++j) {
            out_ptr[i] += prob_vec_ptr[j] * v_block_cache_ptr[j];
          }
        }
      }
    }
  }
  std::free(logits);
}

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                      \
  single_query_cached_kv_attention_impl<T, HEAD_SIZE, BLOCK_SIZE>(             \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, head_mapping_ptr,    \
      scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,       \
      alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads);

template <typename T, int BLOCK_SIZE>
void single_query_cached_kv_attention_cpu_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
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
  int *head_mapping_ptr = reinterpret_cast<int *>(head_mapping.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
  // 32, 160, 192.
  // case 32:
  //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
  //   break;
  case 64:
    LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  // case 160:
  //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE);
  //   break;
  // case 192:
  //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE);
  //   break;
  case 256:
    LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                    \
  single_query_cached_kv_attention_cpu_launcher<T, BLOCK_SIZE>(                \
      out, query, key_cache, value_cache, head_mapping, scale, block_tables,   \
      context_lens, max_context_len, alibi_slopes);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                     \
  switch (block_size) {                                                        \
  /* case 1:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 1);   */                                        \
  /*   break;                        */                                        \
  /* case 2:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 2);   */                                        \
  /*   break;                        */                                        \
  /* case 4:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 4);   */                                        \
  /*   break;                        */                                        \
  case 8:                                                                      \
    CALL_KERNEL_LAUNCHER(T, 8);                                                \
    break;                                                                     \
  case 16:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 16);                                               \
    break;                                                                     \
  case 32:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 32);                                               \
    break;                                                                     \
  /* case 64:                        */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 64);  */                                        \
  /*   break;                        */                                        \
  /* case 128:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 128); */                                        \
  /*   break;                        */                                        \
  /* case 256:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 256); */                                        \
  /*   break;                        */                                        \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }
} // namespace

void single_query_cached_kv_attention_cpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

void single_query_cached_kv_attention_gpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes);

void single_query_cached_kv_attention(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  switch (out.device().type()) {
  case c10::DeviceType::CUDA:
    return single_query_cached_kv_attention_gpu(
        out, query, key_cache, value_cache, head_mapping, scale, block_tables,
        context_lens, block_size, max_context_len, alibi_slopes);
  case c10::DeviceType::CPU:
    return single_query_cached_kv_attention_cpu(
        out, query, key_cache, value_cache, head_mapping, scale, block_tables,
        context_lens, block_size, max_context_len, alibi_slopes);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

void multi_query_cached_kv_attention_cpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  TORCH_CHECK(false, "Unsupported multi_query_cached_kv_attention on cpu.")
}

void multi_query_cached_kv_attention(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  switch (out.device().type()) {
  case c10::DeviceType::CPU:
    return multi_query_cached_kv_attention_cpu(
        out, query, key_cache, value_cache, head_mapping, scale, block_tables,
        context_lens, block_size, max_context_len, alibi_slopes);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_query_cached_kv_attention", &single_query_cached_kv_attention,
        "Compute the attention between an input query and the cached key/value "
        "tensors");
  m.def("multi_query_cached_kv_attention", &multi_query_cached_kv_attention,
        "Compute the attention between multiple input query tensors");
}
