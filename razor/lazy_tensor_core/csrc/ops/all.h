/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class All : public Node {
 public:
  All(const Value& input, std::vector<int64_t> dimensions, bool keep_reduced_dimensions);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<int64_t>& dimensions() const {
    return dimensions_;
  }

  bool keep_reduced_dimensions() const {
    return keep_reduced_dimensions_;
  }

 private:
  std::vector<int64_t> dimensions_;
  bool keep_reduced_dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
