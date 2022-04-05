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

class UpdateSlice : public Node {
 public:
  UpdateSlice(const Value& input, const Value& source,
              lazy_tensors::Span<const int64_t> base_indices);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const {
    return base_indices_;
  }

 private:
  std::vector<int64_t> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
