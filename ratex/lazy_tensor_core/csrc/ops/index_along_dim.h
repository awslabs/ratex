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

class IndexAlongDim : public Node {
 public:
  IndexAlongDim(OpKind op, const ir::Value& buffer, const ir::Value& index, const ir::Value& value,
                int64_t dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t dim() const {
    return dim_;
  }

 private:
  // The dimension along which indexing is applied.
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
