/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ReduceScatter : public Node {
 public:
  ReduceScatter(const Value& input, const Value& token, AllReduceType reduce_type,
                std::vector<std::vector<int64_t>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  AllReduceType reduce_type() const {
    return reduce_type_;
  }

  const std::vector<std::vector<int64_t>>& groups() const {
    return groups_;
  }

 private:
  AllReduceType reduce_type_;
  std::vector<std::vector<int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
