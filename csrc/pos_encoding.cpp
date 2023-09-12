#include <c10/util/Exception.h>
#include <torch/extension.h>

template <typename scalar_t>
void rotary_embedding_impl(
    const int64_t *__restrict__ positions, // [num_tokens]
    scalar_t *__restrict__ query,          // [num_tokens, num_heads, head_size]
    scalar_t *__restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const scalar_t
        *__restrict__ cos_sin_cache, // [max_position, 2, rot_dim // 2]
    const int rot_dim, const int stride, const int num_heads,
    const int num_kv_heads, const int head_size, const int num_tokens) {
#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const scalar_t *cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;
    const int nq = num_heads * embed_dim;
    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int token_head = token_idx * stride + head_idx * head_size;
      for (int j = 0; j < embed_dim; ++j) {
        const int rot_offset = j;
        const int x_index = rot_offset;
        const int y_index = embed_dim + rot_offset;

        const int out_x = token_head + x_index;
        const int out_y = token_head + y_index;

        const scalar_t cos = *(cache_ptr + x_index);
        const scalar_t sin = *(cache_ptr + y_index);

        const scalar_t q_x = query[token_head + x_index];
        const scalar_t q_y = query[token_head + y_index];
        query[out_x] = q_x * cos - q_y * sin;
        query[out_y] = q_y * cos + q_x * sin;

        if (head_idx < num_kv_heads) {
          const scalar_t k_x = key[token_head + x_index];
          const scalar_t k_y = key[token_head + y_index];
          key[out_x] = k_x * cos - k_y * sin;
          key[out_y] = k_y * cos + k_x * sin;
        }
      }
    }
  }
}

void rotary_embedding_cpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache) {
  TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float);

  int num_tokens = query.size(0);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(1) / head_size;
  int num_kv_heads = key.size(1) / head_size;
  int stride = query.stride(0);
  TORCH_CHECK(stride == key.stride(0));

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_impl", [&] {
    rotary_embedding_impl(positions.data_ptr<int64_t>(),
                          query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
                          cos_sin_cache.data_ptr<scalar_t>(), rot_dim, stride,
                          num_heads, num_kv_heads, head_size, num_tokens);
  });
}

void rotary_embedding_gpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox);

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox) {
  switch (positions.device().type()) {
  case c10::DeviceType::CUDA:
    return rotary_embedding_gpu(positions, query, key, head_size, cos_sin_cache,
                                is_neox);
  case c10::DeviceType::CPU:
    TORCH_CHECK(is_neox);
    return rotary_embedding_cpu(positions, query, key, head_size,
                                cos_sin_cache);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &rotary_embedding,
        "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");
}
