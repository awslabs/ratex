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

class Resize : public Node {
 public:
  Resize(const Value& input, std::vector<int64_t> size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

 private:
  std::vector<int64_t> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
