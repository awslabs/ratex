/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2);
NodePtr operator-(const Value& node1, const Value& node2);
NodePtr operator*(const Value& node1, const Value& node2);
NodePtr operator/(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace torch_lazy_tensors
