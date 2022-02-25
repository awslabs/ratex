/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/literal.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Constant : public Node {
 public:
  Constant(lazy_tensors::Literal value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const lazy_tensors::Literal& value() const {
    return value_;
  }

 private:
  lazy_tensors::Literal value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
