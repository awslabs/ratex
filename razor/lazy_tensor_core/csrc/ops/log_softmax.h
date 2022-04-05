/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// IR node for log(softmax) operation.
class LogSoftmax : public Node {
 public:
  LogSoftmax(const Value& input, int64_t dim, c10::optional<at::ScalarType> dtype);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int64_t dim() const {
    return dim_;
  }

  const c10::optional<at::ScalarType>& dtype() const {
    return dtype_;
  }

 private:
  // The dimension along which the result is computed.
  int64_t dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
