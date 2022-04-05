/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

}  // namespace torch_lazy_tensors
