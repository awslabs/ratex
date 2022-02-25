/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace lazy_tensors {

class NNCComputationClient {
 public:
  static at::DeviceType HardwareDeviceType();
};

}  // namespace lazy_tensors
