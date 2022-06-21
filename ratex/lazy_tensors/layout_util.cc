/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensors/layout_util.h"

#include "lazy_tensors/core/platform/hash.h"

namespace lazy_tensors {

size_t LayoutUtil::Hash(const Layout& layout) {
  size_t hash_value = std::hash<size_t>()(0);

  for (int64_t minor_to_major : layout.minor_to_major()) {
    hash_value = Hash64Combine(hash_value, hash<int64_t>()(minor_to_major));
  }

  return hash_value;
}

}  // namespace lazy_tensors
