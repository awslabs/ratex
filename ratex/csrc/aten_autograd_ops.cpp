/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#include "ratex/csrc/aten_autograd_ops.h"

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "ratex/csrc/LazyNativeFunctions.h"
#include "ratex/csrc/aten_cpu_fallback.h"

namespace torch_lazy_tensors {
namespace aten_autograd_ops {

torch::Tensor MaxPool2dAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 torch::Tensor self, torch::IntArrayRef kernel_size,
                                                 torch::IntArrayRef stride,
                                                 torch::IntArrayRef padding,
                                                 torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = FALLBACK_ATEN_OP(max_pool2d_with_indices, self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = LazyTensor::max_pool_nd(bridge::GetLtcTensor(self), /*spatial_dim_count=*/2,
                                         Helpers::I64List(kernel_size), Helpers::I64List(stride),
                                         Helpers::I64List(padding), ceil_mode);
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool2dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = FALLBACK_ATEN_OP(max_pool2d_with_indices_backward, grad_output[0], self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  grad = bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output[0]), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad, undef, undef, undef, undef, undef};
  return grad_inputs;
}

torch::Tensor MaxPool3dAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 torch::Tensor self, torch::IntArrayRef kernel_size,
                                                 torch::IntArrayRef stride,
                                                 torch::IntArrayRef padding,
                                                 torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = FALLBACK_ATEN_OP(max_pool3d_with_indices, self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = LazyTensor::max_pool_nd(bridge::GetLtcTensor(self), /*spatial_dim_count=*/3,
                                         Helpers::I64List(kernel_size), Helpers::I64List(stride),
                                         Helpers::I64List(padding), ceil_mode);
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool3dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = FALLBACK_ATEN_OP(max_pool3d_with_indices_backward, grad_output[0], self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  grad = bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output[0]), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad, undef, undef, undef, undef, undef};
  return grad_inputs;
}

torch::Tensor Dropout::forward(torch::autograd::AutogradContext* ctx, const at::Tensor& input,
                               double p, bool train) {
  ctx->saved_data["p"] = p;
  auto outputs = LazyTensor::dropout(bridge::GetLtcTensor(input), p, train);
  ctx->save_for_backward({bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                          bridge::AtenFromLtcTensor(std::get<2>(outputs))});
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list Dropout::backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_output) {
  auto p = ctx->saved_data["p"];

  auto saved = ctx->get_saved_variables();
  auto mask = bridge::GetLtcTensor(saved[0]);
  auto reserved_space = bridge::GetLtcTensor(saved[1]);
  auto grad = bridge::GetLtcTensor(grad_output[0]);
  auto results = LazyTensor::dropout_backward(grad, mask, reserved_space);

  auto grad_outputs = bridge::AtenFromLtcTensor(results);
  torch::Tensor undef;
  torch::autograd::variable_list grad_result = {grad_outputs, undef, undef};

  return grad_result;
}

}  // namespace aten_autograd_ops
}  // namespace torch_lazy_tensors
