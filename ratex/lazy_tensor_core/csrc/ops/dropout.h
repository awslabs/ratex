/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Dropout : public Node {
 public:
  Dropout(const Value& input, double p);

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
