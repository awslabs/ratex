/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/uniform.h"

#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Uniform::Uniform(const Value& from, const Value& to, const Value& seed,
                 const lazy_tensors::Shape& rng_shape)
    : Node(ir::OpKind(at::aten::uniform), {from, to, seed}, rng_shape,
           /*num_outputs=*/1, lazy_tensors::util::ShapeHash(rng_shape)) {
}

NodePtr Uniform::Clone(OpList operands) const {
  return MakeNode<Uniform>(operands.at(0), operands.at(1), operands.at(2), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
