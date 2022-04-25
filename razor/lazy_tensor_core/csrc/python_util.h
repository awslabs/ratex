/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "absl/types/optional.h"

namespace torch_lazy_tensors {

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

absl::optional<SourceLocation> GetPythonFrameTop();

std::vector<SourceLocation> GetPythonFrames();

std::ostream& operator<<(std::ostream& stream, const std::vector<SourceLocation>& frames);

std::ostream& operator<<(std::ostream& stream, const SourceLocation& frame);
}  // namespace torch_lazy_tensors
