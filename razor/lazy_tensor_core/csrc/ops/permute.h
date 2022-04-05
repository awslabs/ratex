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

class Permute : public Node {
 public:
  Permute(const Value& input, std::vector<int64_t> dims);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const {
    return dims_;
  }

  static lazy_tensors::Shape MakePermuteShape(const lazy_tensors::Shape& source_shape,
                                              lazy_tensors::Span<const int64_t> permutation);

 private:
  // The permutation of dimensions.
  std::vector<int64_t> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
