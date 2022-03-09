/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class RAFAllGather : public Node {
 public:
  RAFAllGather(lazy_tensors::Span<const Value> operands, int64_t dim,
               std::vector<std::vector<int64_t>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t dim() const {
    return dim_;
  }

  const std::vector<std::vector<int64_t>>& groups() const {
    return groups_;
  }

 private:
  int64_t dim_;
  std::vector<std::vector<int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
