/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

extern const OpKindWrapper raf_relay_expr;
extern const OpKindWrapper raf_relay_function;
extern const OpKindWrapper raf_log_softmax_backward_use_in;
extern const OpKindWrapper raf_dropout_backward;

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
