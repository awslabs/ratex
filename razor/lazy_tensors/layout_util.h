/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
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
