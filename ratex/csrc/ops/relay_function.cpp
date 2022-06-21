/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/relay_function.h"
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

lazy_tensors::Shape InferRelayFunction(const Expr& func) {
  FuncType fty;
  if (!func->checked_type_.defined()) {
    LTC_LOG(WARNING) << "func type is not defined and InferType is called to infer the shape. This "
                        "degrades the performance badly.";
    auto f = InferType(func);
    fty = Downcast<FuncType>(f->checked_type());
  } else {
    fty = Downcast<FuncType>(func->checked_type());
  }
  return compiler::raf_backend::ToLTCShape(fty->ret_type);
}

lazy_tensors::Shape GetShape(const Expr& func) {
  lazy_tensors::Shape shape = InferRelayFunction(func);
  return lazy_tensors::Shape(std::vector<lazy_tensors::Shape>({shape}));
}

RelayFunction::RelayFunction(const Expr& func)
    : Node(raf_relay_function, GetShape(func), /*num_outputs=*/1, /*hash_seed=*/202), func_(func) {
}

NodePtr RelayFunction::Clone(OpList operands) const {
  return MakeNode<RelayFunction>(func_);
}

RelayFunction* RelayFunction::Cast(const Node* node) {
  return NodeCast<RelayFunction>(node, raf_relay_function);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
