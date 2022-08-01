/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTATION_CLIENT_DEBUG_MACROS_H_
#define COMPUTATION_CLIENT_DEBUG_MACROS_H_

#include <iostream>

#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/statusor.h"

#define LTC_ERROR() LOG(ERROR)
#define LTC_CHECK(c) CHECK(c)
#define LTC_CHECK_OK(c) TORCH_CHECK(c.ok())
#define LTC_CHECK_EQ(a, b) TORCH_CHECK_EQ(a, b)
#define LTC_CHECK_NE(a, b) TORCH_CHECK_NE(a, b)
#define LTC_CHECK_LE(a, b) TORCH_CHECK_LE(a, b)
#define LTC_CHECK_GE(a, b) TORCH_CHECK_GE(a, b)
#define LTC_CHECK_LT(a, b) TORCH_CHECK_LT(a, b)
#define LTC_CHECK_GT(a, b) TORCH_CHECK_GT(a, b)

template <typename T>
T ConsumeValue(lazy_tensors::StatusOr<T>&& status) {
  LTC_CHECK_OK(status.status());
  return status.ConsumeValueOrDie();
}

#endif  // COMPUTATION_CLIENT_DEBUG_MACROS_H_
