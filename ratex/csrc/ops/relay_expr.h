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

class RelayExpr : public Node {
 public:
  RelayExpr(const std::vector<Value>& inputs);

  // std::string ToString() const override;

  // NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
