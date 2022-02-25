/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raf/registry.h>

namespace razor {

void PrintStack() {
  static auto print_stack = raf::registry::GetPackedFunc("razor.utils.print_stack");
  print_stack();
}

}  // namespace razor
