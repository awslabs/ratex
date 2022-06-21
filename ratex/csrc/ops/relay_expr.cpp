/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/relay_expr.h"
#include "ratex/csrc/ops/raf_ops.h"
#include "ratex/csrc/compiler/utils.h"
#include "client/raf_computation_client.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensors/computation_client/util.h"

#include "raf/value.h"
#include "raf/ir.h"
#include "raf/pass.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

using namespace raf::value;
using namespace raf::pass;
using namespace raf::ir;

lazy_tensors::Shape InferRelayExpr(const std::vector<Value>& inputs) {
  return inputs[0].shape();
}

int64_t GetNumOutputs(const std::vector<Value>& inputs) {
  lazy_tensors::Shape shape = InferRelayExpr(inputs);
  return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
}

RelayExpr::RelayExpr(const std::vector<Value>& inputs)
    : Node(raf_relay_expr, inputs, /*num_outputs=*/GetNumOutputs(inputs), 0) {
  SetShapeDeferred([&]() { return InferRelayExpr(inputs); });
}

// NodePtr RelayExpr::Clone(OpList operands) const {
//   return MakeNode<RelayExpr>(std::vector<Value>(operands.begin(), operands.end()));
// }

// std::string RelayExpr::ToString() const {
//   std::stringstream ss;
//   ss << Node::ToString() << ", closure=" <<  Output(closure_);
//   return ss.str();
// }

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
