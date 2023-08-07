#include <c10/util/Optional.h>
#include <torch/extension.h>

void single_query_cached_kv_attention_cpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  TORCH_CHECK(false, "Unsupported single_query_cached_kv_attention on cpu.")
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
