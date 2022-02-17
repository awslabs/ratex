#include "torch_mnm/csrc/ops/mnm_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

const OpKindWrapper mnm_relay_expr("torch_mnm::relay_expr");
const OpKindWrapper mnm_relay_function("torch_mnm::relay_function");
const OpKindWrapper mnm_log_softmax_backward_use_in("torch_mnm::log_softmax_backward_use_in");
const OpKindWrapper mnm_all_reduce("torch_mnm::all_reduce");
const OpKindWrapper mnm_all_gather("torch_mnm::all_gather");

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
