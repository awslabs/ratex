/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MNMAllGather : public Node {
 public:
  MNMAllGather(lazy_tensors::Span<const Value> operands, lazy_tensors::int64 dim,
               std::vector<std::vector<lazy_tensors::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 dim() const {
    return dim_;
  }

  const std::vector<std::vector<lazy_tensors::int64>>& groups() const {
    return groups_;
  }

 private:
  lazy_tensors::int64 dim_;
  std::vector<std::vector<lazy_tensors::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
