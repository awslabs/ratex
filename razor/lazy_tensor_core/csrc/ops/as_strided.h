/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AsStrided : public Node {
 public:
  AsStrided(const Value& input, std::vector<int64_t> size, std::vector<int64_t> stride,
            int64_t storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

  const std::vector<int64_t>& stride() const {
    return stride_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  static bool StrideIsSupported(const lazy_tensors::Shape& input_shape,
                                lazy_tensors::Span<const int64_t> size,
                                lazy_tensors::Span<const int64_t> stride, int64_t storage_offset);

  static std::vector<int64_t> GetArrayStridePermutation(lazy_tensors::Span<const int64_t> stride,
                                                        lazy_tensors::Span<const int64_t> size);

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
