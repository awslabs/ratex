/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DropoutBackward : public Node {
 public:
  DropoutBackward(const Value& grad_output, const Value& mask, const Value& reserve_space);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const double p() const {
    return p_;
  };

 private:
  double p_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
