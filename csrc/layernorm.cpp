#include <ATen/Dispatch.h>
#include <torch/extension.h>

void rms_norm_gpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon);

namespace {
template <typename scalar_t>
void rms_norm_cpu_impl(scalar_t *__restrict__ out,
                       const scalar_t *__restrict__ input,
                       const scalar_t *__restrict__ weight, const float epsilon,
                       const int num_tokens, const int hidden_size) {
#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    float variance = 0.0f;
    auto input_p = input + i * hidden_size;
    auto output_p = out + i * hidden_size;
    // omp simd reduction is useless
    for (int j = 0; j < hidden_size; ++j) {
      const float x = (float)input_p[j];
      variance += x * x;
    }
    float s_variance = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);
    // TBD: gcc compiler only use 128-bit of YMM for multiplications, can't
    // fully leverage AVX2 TBD: SSE doesn't lead improvement for this loop,
    // maybe due to the constant s_variance.
    for (int j = 0; j < hidden_size; ++j) {
      float x = (float)input_p[j];
      output_p[j] = ((scalar_t)(x * s_variance)) * weight[j];
    }
  }
}
} // namespace

void rms_norm_cpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float);

  int num_tokens = input.size(0);
  int hidden_size = input.size(1);
  int num_tokens_stride = input.stride(0);
  int hidden_size_stride = input.stride(1);
  TORCH_CHECK_EQ(hidden_size_stride, 1);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_cpu_impl", [&] {
    rms_norm_cpu_impl(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                      weight.data_ptr<scalar_t>(), epsilon, num_tokens,
                      hidden_size);
  });
}

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight,
              float epsilon) {
  switch (weight.device().type()) {
  case c10::DeviceType::CUDA:
    return rms_norm_gpu(out, input, weight, epsilon);
  case c10::DeviceType::CPU:
    return rms_norm_cpu(out, input, weight, epsilon);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}
