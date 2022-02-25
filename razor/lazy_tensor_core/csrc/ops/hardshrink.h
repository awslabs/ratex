/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Hardshrink : public Node {
 public:
  Hardshrink(const Value& input, const at::Scalar& lambda);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  at::Scalar lambda() const {
    return lambda_;
  }

 private:
  at::Scalar lambda_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
