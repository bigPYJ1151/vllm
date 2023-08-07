#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks_cpu(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping) {
  // Swap on CPU is unnecessary.
  return;
}

void copy_blocks_cpu(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping) {
  TORCH_CHECK(false, "Unsupported copy_blocks on cpu.")
}

void reshape_and_cache_cpu(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping) {
  TORCH_CHECK(false, "Unsupported reshape_and_cache on cpu.")
}

void gather_cached_kv_cpu(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping) {
  // gather_cached_kv is on;y used for testing.
  TORCH_CHECK(false, "Unsupported gather_cached_kv on cpu.")
  return;
}

void swap_blocks_gpu(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping);

void copy_blocks_gpu(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache_gpu(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping);

void gather_cached_kv_gpu(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping);

void swap_blocks(torch::Tensor &src, torch::Tensor &dst,
                 const std::map<int64_t, int64_t> &block_mapping) {
  switch (src.device().type()) {
  case c10::DeviceType::CUDA:
    return swap_blocks_gpu(src, dst, block_mapping);
  case c10::DeviceType::CPU:
    return swap_blocks_cpu(src, dst, block_mapping);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

void copy_blocks(std::vector<torch::Tensor> &key_caches,
                 std::vector<torch::Tensor> &value_caches,
                 const std::map<int64_t, std::vector<int64_t>> &block_mapping) {
  switch (key_caches.front().device().type()) {
  case c10::DeviceType::CUDA:
    return copy_blocks_gpu(key_caches, value_caches, block_mapping);
  case c10::DeviceType::CPU:
    return copy_blocks_cpu(key_caches, value_caches, block_mapping);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

void reshape_and_cache(torch::Tensor &key, torch::Tensor &value,
                       torch::Tensor &key_cache, torch::Tensor &value_cache,
                       torch::Tensor &slot_mapping) {
  switch (key.device().type()) {
  case c10::DeviceType::CUDA:
    return reshape_and_cache_gpu(key, value, key_cache, value_cache,
                                 slot_mapping);
  case c10::DeviceType::CPU:
    return reshape_and_cache_cpu(key, value, key_cache, value_cache,
                                 slot_mapping);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

void gather_cached_kv(torch::Tensor &key, torch::Tensor &value,
                      torch::Tensor &key_cache, torch::Tensor &value_cache,
                      torch::Tensor &slot_mapping) {
  switch (key.device().type()) {
  case c10::DeviceType::CUDA:
    return gather_cached_kv_gpu(key, value, key_cache, value_cache,
                                slot_mapping);
  case c10::DeviceType::CPU:
    return gather_cached_kv_cpu(key, value, key_cache, value_cache,
                                slot_mapping);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swap_blocks", &swap_blocks,
        "Swap in (out) the cache blocks from src to dst");
  m.def("copy_blocks", &copy_blocks, "Copy the cache blocks from src to dst");
  m.def("reshape_and_cache", &reshape_and_cache,
        "Reshape the key and value tensors and cache them");
  m.def("gather_cached_kv", &gather_cached_kv,
        "Gather key and value from the cache into contiguous QKV tensors");
}
