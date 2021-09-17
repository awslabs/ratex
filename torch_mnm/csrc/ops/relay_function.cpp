#include "torch_mnm/csrc/ops/relay_function.h"
#include "torch_mnm/csrc/ops/mnm_ops.h"
#include "torch_mnm/csrc/compiler/utils.h"
#include "mnm_client/mnm_computation_client.h"

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

lazy_tensors::Shape InferRelayFunction(const Expr& func) {
  Expr f = InferType(func);
  auto fty = Downcast<FuncType>(f->checked_type());
  return compiler::mnm_backend::ToLTCShape(fty->ret_type);
}

lazy_tensors::Shape GetShape(const Expr& func) {
  lazy_tensors::Shape shape = InferRelayFunction(func);
  return lazy_tensors::Shape(std::vector<lazy_tensors::Shape>({shape}));
}

RelayFunction::RelayFunction(const Expr& func)
    : Node(mnm_relay_function, GetShape(func), /*num_outputs=*/1, /*hash_seed=*/202),
    func_(func) { }

NodePtr RelayFunction::Clone(OpList operands) const {
  return MakeNode<RelayFunction>(func_);
}

RelayFunction* RelayFunction::Cast(const Node* node) {
  return NodeCast<RelayFunction>(node, mnm_relay_function);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
