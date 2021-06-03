#pragma once

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

NodeLowering* GetMNMNodeLowering();
std::unique_ptr<NodeLowering> CreateMNMNodeLowering(ir::LoweringContext* loctx);

}  // namespace compiler
}  // namespace torch_lazy_tensors
