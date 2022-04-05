/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ThresholdBackward : public Node {
 public:
  ThresholdBackward(const Value& grad_output, const Value& input, float threshold);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  float threshold() const {
    return threshold_;
  }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
