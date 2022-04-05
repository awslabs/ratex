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

class SVD : public Node {
 public:
  SVD(const Value& input, bool some, bool compute_uv);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool some() const {
    return some_;
  }

  bool compute_uv() const {
    return compute_uv_;
  }

 private:
  bool some_;
  bool compute_uv_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
