/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an LazyTensor.
class LTCTensorImpl : public c10::TensorImpl {
 public:
  explicit LTCTensorImpl(LazyTensor tensor);

  LazyTensor& tensor() {
    return tensor_;
  }

  void set_tensor(LazyTensor lazy_tensor);

  void force_refresh_sizes() {
    generation_ = 0;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter, bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes_custom() const override;

  at::IntArrayRef strides_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

  void MarkAsInteropView() {
    is_interop_view_ = true;
  }

  bool IsInteropView() const {
    return is_interop_view_;
  }

  static void AtenInitialize();

 private:
  void SetupSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const LazyTensor& tensor);

  LazyTensor tensor_;
  size_t generation_ = 0;
  bool is_interop_view_ = false;
};

}  // namespace torch_lazy_tensors
