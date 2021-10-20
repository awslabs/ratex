#pragma once

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

BackendImplInterface* GetXlaBackendImpl();

}  // namespace compiler
}  // namespace torch_lazy_tensors
