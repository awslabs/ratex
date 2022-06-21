/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexPut : public Node {
 public:
  IndexPut(const ir::Value& base, const ir::Value& indices, int64_t start_dim,
           const ir::Value& values, bool accumulate);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t start_dim() const {
    return start_dim_;
  }

  bool accumulate() const {
    return accumulate_;
  }

 private:
  // The dimension number at which indexing starts.
  int64_t start_dim_;
  // Whether to accumulate instead of set.
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
