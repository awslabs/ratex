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

class MatMul : public Node {
 public:
  MatMul(const Value& input, const Value& other, std::vector<int64_t> a_shape,
         std::vector<int64_t> b_shape);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  std::vector<int64_t> a_shape() const {
    return a_shape_;
  }

  std::vector<int64_t> b_shape() const {
    return b_shape_;
  }

 private:
  std::vector<int64_t> a_shape_;
  std::vector<int64_t> b_shape_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors