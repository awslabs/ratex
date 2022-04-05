/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GenericSlice : public Node {
 public:
  GenericSlice(const Value& input, lazy_tensors::Span<const int64_t> base_indices,
               lazy_tensors::Span<const int64_t> sizes);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const {
    return base_indices_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
