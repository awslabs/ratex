/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace torch_lazy_tensors {
namespace fn_tracker {

#define LTC_FN_TRACK(level) torch_lazy_tensors::fn_tracker::TrackFunction(__FUNCTION__, level)

void TrackFunction(const char* tag, int level);

}  // namespace fn_tracker
}  // namespace torch_lazy_tensors
