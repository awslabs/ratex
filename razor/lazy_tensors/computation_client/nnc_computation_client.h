/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace lazy_tensors {

class NNCComputationClient {
 public:
  static at::DeviceType HardwareDeviceType();
};

}  // namespace lazy_tensors
