/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Node for the backward_layer_norm operator.
class NativeLayerNormBackward : public Node {
 public:
  NativeLayerNormBackward(const Value& grad_out, const Value& input, std::vector<int64_t> normalized_shape,
                          const Value& mean, const Value& rstd, const Value& weight, const Value& bias);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  std::vector<int64_t> normalized_shape() const {
    return normalized_shape_;
  }

  private:
  std::vector<int64_t> normalized_shape_;

};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
