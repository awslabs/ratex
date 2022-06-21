/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Expand : public Node {
 public:
  Expand(const Value& input, std::vector<int64_t> size);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<int64_t>& size() const {
    return size_;
  };

 private:
  std::vector<int64_t> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
