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

class Put : public Node {
 public:
  Put(const Value& input, const Value& index, const Value& source, bool accumulate);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool accumulate() const {
    return accumulate_;
  }

 private:
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
