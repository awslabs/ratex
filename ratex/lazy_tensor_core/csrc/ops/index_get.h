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

class IndexGet : public Node {
 public:
  IndexGet(const ir::Value& base, const ir::Value& indices, int64_t start_dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t start_dim() const {
    return start_dim_;
  }

 private:
  // The dimension number at which indexing starts.
  int64_t start_dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
