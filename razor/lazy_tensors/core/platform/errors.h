/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensors/core/platform/macros.h"

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                          \
  do {                                                   \
    ::lazy_tensors::Status _status = (__VA_ARGS__);      \
    if (TF_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)
