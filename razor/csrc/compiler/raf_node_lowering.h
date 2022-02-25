/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

NodeLowering* GetRAFNodeLowering();
std::unique_ptr<NodeLowering> CreateRAFNodeLowering(ir::LoweringContext* loctx);

}  // namespace compiler
}  // namespace torch_lazy_tensors
