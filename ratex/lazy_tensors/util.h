/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensors/span.h"

namespace lazy_tensors {

inline lazy_tensors::Span<const int64_t> AsInt64Slice(lazy_tensors::Span<const int64_t> slice) {
  return slice;
}

}  // namespace lazy_tensors
