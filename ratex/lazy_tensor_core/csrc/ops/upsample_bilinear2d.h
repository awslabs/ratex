/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpsampleBilinear : public Node {
 public:
  UpsampleBilinear(const Value& input, std::vector<int64_t> output_size, bool align_corners);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const {
    return output_size_;
  }

  bool align_corners() const {
    return align_corners_;
  }

 private:
  std::vector<int64_t> output_size_;
  bool align_corners_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
