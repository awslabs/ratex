/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/raf_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

const OpKindWrapper raf_relay_expr("ratex::relay_expr");
const OpKindWrapper raf_relay_function("ratex::relay_function");
const OpKindWrapper raf_log_softmax_backward_use_in("ratex::log_softmax_backward_use_in");
const OpKindWrapper raf_dropout_backward("razor::dropout_backward");

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
