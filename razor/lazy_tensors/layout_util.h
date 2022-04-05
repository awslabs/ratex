/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensors/layout.h"

namespace lazy_tensors {

// Namespaced collection of (static) Layout utilities.
class LayoutUtil {
 public:
  // Compute a hash for `layout`.
  static size_t Hash(const Layout& layout);
};

}  // namespace lazy_tensors
