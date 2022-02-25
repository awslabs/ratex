/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mnm/registry.h>

namespace razor {

void PrintStack() {
  static auto print_stack = mnm::registry::GetPackedFunc("razor.utils.print_stack");
  print_stack();
}

}  // namespace razor
