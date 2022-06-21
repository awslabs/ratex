/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"
#include "raf/value.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class RelayFunction : public Node {
 public:
  RelayFunction(const raf::ir::Expr& func);

  NodePtr Clone(OpList operands) const override;

  const raf::ir::Expr& func() const {
    return func_;
  }

  static RelayFunction* Cast(const Node* node);

 private:
  raf::ir::Expr func_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
