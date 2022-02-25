/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
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
