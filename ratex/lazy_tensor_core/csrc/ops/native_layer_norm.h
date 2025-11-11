/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
//Node for the native_layer_norm operator.
class NativeLayerNorm : public Node {
 public:
  NativeLayerNorm(const Value& input,std::vector<int64_t> normalized_shape, const Value& weight, const Value& bias, double eps);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  std::vector<int64_t> normalized_shape() const {
    return normalized_shape_;
  }

  double eps() const {
    return eps_;
  }


 private:
  std::vector<int64_t> normalized_shape_;
  double eps_;

};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
