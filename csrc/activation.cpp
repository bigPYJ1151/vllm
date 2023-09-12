#include <torch/extension.h>

template <typename T> __inline__ T silu_cpu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t>
void silu_and_mul_cpu_impl(int num_tokens, int d, scalar_t *__restrict__ input,
                           scalar_t *__restrict__ output) {
#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < d; j++) {
      const int start = i * 2 * d;
      const scalar_t x = input[start + j];
      const scalar_t y = input[start + d + j];
      output[i * d + j] = silu_cpu(x) * y;
    }
  }
}

void silu_and_mul_cpu(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float);
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rotary_embedding_neox_impl", [&] {
        silu_and_mul_cpu_impl(num_tokens, d, input.data_ptr<scalar_t>(),
                              out.data_ptr<scalar_t>());
      });
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

void gelu_new(torch::Tensor &out, torch::Tensor &input);

void gelu_fast(torch::Tensor &out, torch::Tensor &input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
  m.def("gelu_new", &gelu_new, "GELU implementation used in GPT-2.");
  m.def("gelu_fast", &gelu_fast, "Approximate GELU implementation.");
}
