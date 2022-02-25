/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "razor/csrc/ops/mnm_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

const OpKindWrapper mnm_relay_expr("razor::relay_expr");
const OpKindWrapper mnm_relay_function("razor::relay_function");
const OpKindWrapper mnm_log_softmax_backward_use_in("razor::log_softmax_backward_use_in");
const OpKindWrapper mnm_all_reduce("razor::all_reduce");
const OpKindWrapper mnm_all_gather("razor::all_gather");

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
