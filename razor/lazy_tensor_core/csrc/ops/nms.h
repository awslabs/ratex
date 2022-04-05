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
namespace ops {

class Nms : public Node {
 public:
  Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
      const Value& iou_threshold, int64_t output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int64_t output_size() const {
    return output_size_;
  }

 private:
  int64_t output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
