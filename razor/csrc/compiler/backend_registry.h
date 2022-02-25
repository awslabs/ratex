/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

class BackendImplRegistry {
 public:
  static BackendImplRegistry* Get();

  BackendImplInterface* GetBackendImpl();
  BackendImplRegistry* AddBackendImpl(BackendImplInterface* backend_impl, int priority);

 private:
  std::map<BackendImplInterface*, int> backend_impls_;
};

BackendImplRegistry* GetBackendImplRegistry();

}  // namespace compiler
}  // namespace torch_lazy_tensors
