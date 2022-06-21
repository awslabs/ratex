/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <c10/util/Logging.h>

#define RATEX_VLOG(n) ::c10::MessageLogger((char*)__FILE__, __LINE__, -n).stream()

namespace c10 {
namespace detail {

void setLogLevelFlag(int logging_level);

}  // namespace detail
}  // namespace c10