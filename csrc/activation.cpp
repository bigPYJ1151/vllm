#include <torch/extension.h>

void silu_and_mul_cpu(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(false, "Unsupported silu_and_mul on cpu.")
}

void silu_and_mul_gpu(torch::Tensor &out, torch::Tensor &input);

void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  switch (out.device().type()) {
  case c10::DeviceType::CUDA:
    return silu_and_mul_gpu(out, input);
  case c10::DeviceType::CPU:
    return silu_and_mul_cpu(out, input);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  m.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  m.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");
}
