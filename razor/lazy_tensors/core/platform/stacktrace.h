/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "lazy_tensors/computation_client/ltc_logging.h"

namespace lazy_tensors {

inline std::string CurrentStackTrace() {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace lazy_tensors
