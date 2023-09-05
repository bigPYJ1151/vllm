#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks_cpu(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping) {
  // Swap on CPU is unnecessary.
  return;
}

template <typename scalar_t>
void copy_blocks_cpu_impl(std::vector<torch::Tensor> &key_caches,
                          std::vector<torch::Tensor> &value_caches,
                          const std::vector<std::pair<int, int>> mapping_pairs,
                          const int element_num_per_block,
                          const int layer_num) {
  const size_t pair_num = mapping_pairs.size();
  const size_t block_bytes = sizeof(scalar_t) * element_num_per_block;
#pragma omp parallel for collapse(2)
  for (int layer = 0; layer < layer_num; ++layer) {
    for (size_t pair = 0; pair < pair_num; ++pair) {
      int source_offset = element_num_per_block * mapping_pairs[pair].first;
      int target_offset = element_num_per_block * mapping_pairs[pair].second;
      scalar_t *key_cache_ptr = key_caches[layer].data_ptr<scalar_t>();
      scalar_t *source_ptr = key_cache_ptr + source_offset;
      scalar_t *target_ptr = key_cache_ptr + target_offset;
      std::memcpy(target_ptr, source_ptr, block_bytes);

      scalar_t *value_cache_ptr = value_caches[layer].data_ptr<scalar_t>();
      source_ptr = value_cache_ptr + source_offset;
      target_ptr = value_cache_ptr + target_offset;
      std::memcpy(target_ptr, source_ptr, block_bytes);
    }
  }
}

void copy_blocks_cpu(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }

  std::vector<std::pair<int, int>> mapping_pairs;
  mapping_pairs.reserve(block_mapping.size());
  for (const auto &pair : block_mapping) {
    for (const auto &dst : pair.second) {
      mapping_pairs.emplace_back(pair.first, dst);
    }
  }

  const int element_num_per_block = key_caches[0][0].numel();
  AT_DISPATCH_FLOATING_TYPES(
      key_caches[0].scalar_type(), "copy_blocks_cpu_impl", [&] {
        copy_blocks_cpu_impl<scalar_t>(key_caches, value_caches, mapping_pairs,
                                       element_num_per_block, num_layers);
      });
}

template <typename scalar_t>
void reshape_and_cache_cpu_impl(
    const scalar_t *__restrict__ key, const scalar_t *__restrict__ value,
    scalar_t *__restrict__ key_cache, scalar_t *__restrict__ value_cache,
    const int *__restrict__ slot_mapping, const int num_tokens,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int block_elem_num = num_heads * head_size * block_size;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      int src_key_head_idx = token_idx * key_stride + head_idx * head_size;
      int src_value_head_idx = token_idx * value_stride + head_idx * head_size;
      const scalar_t *src_key_head_ptr = key + src_key_head_idx;
      const scalar_t *src_value_head_ptr = value + src_value_head_idx;

      const int slot_idx = slot_mapping[token_idx];
      const int block_index = slot_idx / block_size;
      const int block_offset = slot_idx % block_size;
      scalar_t *target_key_head_ptr = key_cache + block_elem_num * block_index +
                                      head_idx * block_size * head_size;
      scalar_t *target_value_head_ptr = value_cache +
                                        block_elem_num * block_index +
                                        head_idx * block_size * head_size;

      for (int src_key_idx = 0; src_key_idx < head_size; src_key_idx += x) {
        const int target_offset = src_key_idx * block_size + block_offset * x;
        for (int i = 0; i < x; ++i) {
          target_key_head_ptr[target_offset + i] =
              src_key_head_ptr[src_key_idx + i];
        }
      }

      for (int src_value_idx = 0; src_value_idx < head_size; ++src_value_idx) {
        const int target_offset = src_value_idx * block_size + block_offset;
        target_value_head_ptr[target_offset] =
            src_value_head_ptr[src_value_idx];
      }
    }
  }
}

void reshape_and_cache_cpu(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  AT_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "reshape_and_cache_cpu_impl", [&] {
        reshape_and_cache_cpu_impl<scalar_t>(
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int>(), num_tokens, key_stride, value_stride,
            num_heads, head_size, block_size, x);
      });
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
