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

class MaskedFill : public Node {
 public:
  MaskedFill(const Value& input, const Value& mask, const at::Scalar& value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  at::Scalar value() const {
    return value_;
  }

 private:
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
