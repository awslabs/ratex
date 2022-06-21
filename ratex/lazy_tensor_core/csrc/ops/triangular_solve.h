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

class TriangularSolve : public Node {
 public:
  TriangularSolve(const Value& rhs, const Value& lhs, bool left_side, bool lower, bool transpose,
                  bool unit_diagonal);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool left_side() const {
    return left_side_;
  }

  bool lower() const {
    return lower_;
  }

  bool transpose() const {
    return transpose_;
  }

  bool unit_diagonal() const {
    return unit_diagonal_;
  }

 private:
  bool left_side_;
  bool lower_;
  bool transpose_;
  bool unit_diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
