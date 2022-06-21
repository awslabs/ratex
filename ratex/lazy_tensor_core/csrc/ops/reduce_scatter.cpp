/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/reduce_scatter.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

std::vector<Value> GetOperandList(lazy_tensors::Span<const Value> operands, const Value& token) {
  std::vector<Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(token);
  return operand_list;
}

}  // namespace

ReduceScatter::ReduceScatter(lazy_tensors::Span<const Value> operands, const Value& token,
                             AllReduceType reduce_type, std::vector<std::vector<int64_t>> groups)
    : Node(ltc_reduce_scatter, GetOperandList(operands, token),
           /*num_outputs=*/2,
           lazy_tensors::util::MHash(lazy_tensors::util::GetEnumValue(reduce_type), groups)),
      reduce_type_(reduce_type),
      groups_(std::move(groups)) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ReduceScatter::Clone(OpList operands) const {
  std::vector<Value> operand_list(operands.begin(), operands.end() - 1);
  return MakeNode<ReduceScatter>(operand_list, operands.back(), reduce_type_, groups_);
}

std::string ReduceScatter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", reduce_type=" << lazy_tensors::util::GetEnumValue(reduce_type_)
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
