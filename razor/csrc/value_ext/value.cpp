/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./value.h"

namespace raf {
namespace value {

using namespace raf::ir;

ClosureValueExt ClosureValueExt::make(ir::Map<ir::Var, Value> env, IRModule mod, GlobalVar gvar) {
  auto ptr = make_object<ClosureValueExtObj>();
  ptr->env = env;
  ptr->mod = mod;
  ptr->gvar = gvar;
  return ClosureValueExt(ptr);
}

RAF_REGISTER_OBJECT_REFLECT(ClosureValueExtObj);

}  // namespace value
}  // namespace raf
