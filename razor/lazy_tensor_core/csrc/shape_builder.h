/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(lazy_tensors::PrimitiveType type) : type_(type) {
  }

  ShapeBuilder& Add(const lazy_tensors::Shape& shape, int64_t dim);

  ShapeBuilder& Add(const lazy_tensors::Shape& shape, lazy_tensors::Span<const int64_t> dimensions);

  ShapeBuilder& Add(int64_t size);

  lazy_tensors::Shape Build() const;

 private:
  struct ShapeDim {
    const lazy_tensors::Shape* shape = nullptr;
    int64_t dim_or_size = -1;
  };

  lazy_tensors::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace torch_lazy_tensors
