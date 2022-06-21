/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/compiler/backend_registry.h"

#include "lazy_tensors/computation_client/ltc_logging.h"

namespace torch_lazy_tensors {
namespace compiler {

BackendImplRegistry* BackendImplRegistry::Get() {
  static BackendImplRegistry backend_impl_registry;
  return &backend_impl_registry;
}

BackendImplInterface* BackendImplRegistry::GetBackendImpl() {
  int priority = -1;
  BackendImplInterface* backend_impl = nullptr;
  for (const auto& kv : backend_impls_) {
    if (kv.second > priority) {
      backend_impl = kv.first;
      priority = kv.second;
    }
  }
  LTC_CHECK(backend_impl);
  return backend_impl;
}

BackendImplRegistry* BackendImplRegistry::AddBackendImpl(BackendImplInterface* backend_impl,
                                                         int priority) {
  if (backend_impls_.find(backend_impl) != backend_impls_.end()) {
    LTC_LOG(FATAL) << "BackendImpl already exists!";
  }
  backend_impls_[backend_impl] = priority;
  return this;
}

BackendImplRegistry* GetBackendImplRegistry() {
  return BackendImplRegistry::Get();
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
