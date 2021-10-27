#include "torch_mnm/csrc/ops/relay_expr.h"
#include "torch_mnm/csrc/ops/mnm_ops.h"
#include "torch_mnm/csrc/compiler/utils.h"
#include "client/mnm_computation_client.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensors/computation_client/util.h"

#include "mnm/value.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

using namespace mnm::value;
using namespace mnm::pass;
using namespace mnm::ir;

lazy_tensors::Shape InferRelayExpr(const std::vector<Value>& inputs) {
  return inputs[0].shape();
}

lazy_tensors::int64 GetNumOutputs(const std::vector<Value>& inputs) {
  lazy_tensors::Shape shape = InferRelayExpr(inputs);
  return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
}

RelayExpr::RelayExpr(const std::vector<Value>& inputs)
    : Node(mnm_relay_expr, inputs, /*num_outputs=*/GetNumOutputs(inputs), 0) {
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
