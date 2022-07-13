/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Embedding : public Node {
 public:
  Embedding(const Value& weight, const Value& indices);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors