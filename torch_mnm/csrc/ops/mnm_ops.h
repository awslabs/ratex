#pragma once

#include <mutex>
#include <string>

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

extern const OpKindWrapper mnm_relay_expr;
extern const OpKindWrapper mnm_relay_function;
extern const OpKindWrapper mnm_log_softmax_backward_use_in;
extern const OpKindWrapper mnm_all_reduce;
extern const OpKindWrapper mnm_all_gather;

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
