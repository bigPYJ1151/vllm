#include <c10/util/Exception.h>
#include <torch/extension.h>

void rotary_embedding_cpu(torch::Tensor &positions, torch::Tensor &query,
                               torch::Tensor &key, int head_size,
                               torch::Tensor &cos_sin_cache) {
  TORCH_CHECK(false, "Unsupported rotary_embedding_neox on cpu.")
}


void rotary_embedding_gpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox);

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox) {
  switch (positions.device().type()) {
  case c10::DeviceType::CUDA:
    return rotary_embedding_gpu(positions, query, key, head_size,
                                     cos_sin_cache, is_neox);
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
