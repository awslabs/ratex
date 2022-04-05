/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ReflectionPad2d : public Node {
 public:
  ReflectionPad2d(const Value& input, std::vector<int64_t> padding);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const {
    return padding_;
  }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
