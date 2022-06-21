/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ConstantPadNd : public Node {
 public:
  ConstantPadNd(const Value& input, std::vector<int64_t> pad, const at::Scalar& value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const at::Scalar& value() const {
    return value_;
  }

  const std::vector<int64_t>& pad() const {
    return pad_;
  }

 private:
  std::vector<int64_t> pad_;
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
