/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/masked_select.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const Value& input) {
  const lazy_tensors::Shape& input_shape = input.shape();
  int64_t input_elements = lazy_tensors::ShapeUtil::ElementsIn(input_shape);
  lazy_tensors::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  lazy_tensors::Shape result_shape =
      lazy_tensors::ShapeUtil::MakeShape(input_shape.element_type(), {input_elements});
  result_shape.set_dynamic_dimension(0, true);
  return lazy_tensors::ShapeUtil::MakeTupleShape(
      {result_shape, lazy_tensors::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

MaskedSelect::MaskedSelect(const Value& input, const Value& mask)
    : Node(ir::OpKind(at::aten::masked_select), {input, mask}, NodeOutputShape(input),
           /*num_outputs=*/2) {
}

NodePtr MaskedSelect::Clone(OpList operands) const {
  return MakeNode<MaskedSelect>(operands.at(0), operands.at(1));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
