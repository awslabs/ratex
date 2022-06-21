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

class DiscreteUniform : public Node {
 public:
  DiscreteUniform(const Value& from, const Value& to, const Value& seed,
                  const lazy_tensors::Shape& rng_shape);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
