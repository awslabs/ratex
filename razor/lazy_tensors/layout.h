/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>

#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

class Tile {};

class Layout {
 public:
  int64_t minor_to_major(int index) const {
    return minor_to_major_.at(index);
  }

  lazy_tensors::Span<const int64_t> minor_to_major() const {
    return minor_to_major_;
  }

  std::vector<int64_t>* mutable_minor_to_major() {
    return &minor_to_major_;
  }

  Layout& add_minor_to_major(int64_t value) {
    minor_to_major_.push_back(value);
    return *this;
  }

 private:
  std::vector<int64_t> minor_to_major_;
};

}  // namespace lazy_tensors
