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

class View : public Node {
 public:
  View(const Value& input, std::vector<int64_t> output_size);

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const {
    return output_size_;
  }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
