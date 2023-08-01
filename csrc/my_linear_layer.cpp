#include <iostream>
#include <mkl_cblas.h>
#include <torch/torch.h>

int main() {
  std::vector<float> x_data = {1, 2, 3};
  std::vector<float> y_data = {1, 2, 3};
  auto x = torch::tensor(torch::ArrayRef<float>(x_data));
  auto y = torch::tensor(torch::ArrayRef<float>(y_data));

  auto sum =
      cblas_sdot(x.size(0), x.data_ptr<float>(), 1, y.data_ptr<float>(), 1);
  std::cout << sum << std::endl;
}