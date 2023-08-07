#include <torch/extension.h>

void rms_norm_gpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon);

void rms_norm_cpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon) {
  TORCH_CHECK(false, "Unsupported rms_norm on cpu.")
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
