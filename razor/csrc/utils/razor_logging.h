/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Below are different implementations for glog and non-glog cases.
#ifdef C10_USE_GLOG
#include <c10/util/logging_is_google_glog.h>
#else  // !C10_USE_GLOG
#include <c10/util/logging_is_not_google_glog.h>
#endif  // C10_USE_GLOG

#define RAZOR_VLOG(n) \
  if (-n >= CAFFE2_LOG_THRESHOLD) ::c10::MessageLogger((char*)__FILE__, __LINE__, -n).stream()