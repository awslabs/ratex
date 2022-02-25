/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Node for the lower triangular part of a matrix (2-D tensor) or batch of
// matrices input.
class Tril : public Node {
 public:
  Tril(const Value& input, lazy_tensors::int64 diagonal);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 diagonal() const {
    return diagonal_;
  }

 private:
  lazy_tensors::int64 diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
