/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "lazy_tensors/computation_client/types.h"
#include "lazy_tensors/span.h"

// Collection of lowerings for operations which only involve some form of data
// movement and no computation.
namespace torch_lazy_tensors {

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<int64_t> GetCompleteShape(lazy_tensors::Span<const int64_t> output_sizes,
                                      lazy_tensors::Span<const int64_t> input_sizes);

std::vector<int64_t> BuildSqueezedDimensions(lazy_tensors::Span<const int64_t> dimensions,
                                             int64_t squeeze_dim);

std::vector<int64_t> BuildUnsqueezeDimensions(lazy_tensors::Span<const int64_t> dimensions,
                                              int64_t dim);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(int64_t dim_size, lazy_tensors::Span<const int64_t> split_sizes);

}  // namespace torch_lazy_tensors
