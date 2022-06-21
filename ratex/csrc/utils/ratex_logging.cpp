/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex_logging.h"

namespace c10 {
namespace detail {

void setLogLevelFlag(int logging_level) {
  FLAGS_caffe2_log_level = logging_level;
}

}  // namespace detail
}  // namespace c10