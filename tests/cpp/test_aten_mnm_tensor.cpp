#include <gtest/gtest.h>
#include <torch/torch.h>

#include <iostream>

#include "cpp_test_util.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "torch_mnm/csrc/compiler/backend_impl.h"
#include "torch_mnm_test.h"

namespace torch_lazy_tensors {
namespace cpp_test {
namespace {

class AtenMNMTensorTest : public AtenMNMTensorTestBase {};

compiler::BackendRegistrar g_registrar(compiler::GetMNMBackendImpl());

}  // namespace

TEST_F(AtenMNMTensorTest, TestRelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::relu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor mnm_input = CopyToDevice(input, device);
    torch::Tensor mnm_output = torch::relu(mnm_input);
    AllClose(output, mnm_output);
  });
}

TEST_F(AtenMNMTensorTest, TestReluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor mnm_input = CopyToDevice(input, device);
    torch::Tensor output = torch::relu_(input);
    torch::Tensor mnm_output = torch::relu_(mnm_input);
    AllClose(output, mnm_output);
    AllClose(input, mnm_output);
  });
}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
