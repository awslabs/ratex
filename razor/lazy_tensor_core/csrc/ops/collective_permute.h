/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class CollectivePermute : public Node {
 public:
  CollectivePermute(const Value& input, const Value& token,
                    std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
