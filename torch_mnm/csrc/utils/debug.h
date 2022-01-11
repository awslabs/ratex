#pragma once

#include <mnm/registry.h>

namespace torch_mnm {

void PrintStack() {
  static auto print_stack = mnm::registry::GetPackedFunc("torch_mnm.utils.print_stack");
  print_stack();
}

}  // namespace torch_mnm
