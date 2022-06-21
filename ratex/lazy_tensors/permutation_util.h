/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "absl/strings/str_join.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

std::vector<int64_t> InversePermutation(lazy_tensors::Span<const int64_t> input_permutation);

bool IsPermutation(lazy_tensors::Span<const int64_t> permutation);

bool IsIdentityPermutation(lazy_tensors::Span<const int64_t> permutation);

template <typename Container>
inline std::vector<typename Container::value_type> PermuteInverse(
    const Container& input, lazy_tensors::Span<const int64_t> permutation) {
  using T = typename Container::value_type;
  lazy_tensors::Span<const T> data(input);
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

}  // namespace lazy_tensors
