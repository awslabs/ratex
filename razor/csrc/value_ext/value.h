/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tvm/ir/module.h"
#include "mnm/ir.h"
#include "mnm/value.h"

namespace mnm {
namespace value {

/*!
 * \brief An object representing an extended closure.
 */
class ClosureValueExtObj final : public ValueObj {
 public:
  ir::Map<ir::Var, Value> env;
  ir::IRModule mod;
  ir::GlobalVar gvar;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_env", &env);
    v->Visit("_mod", &mod);
  }

  static constexpr const char* _type_key = "mnm.value.ClosureValueExt";
  MNM_FINAL_OBJECT(ClosureValueExtObj, ValueObj);
};

/*! \brief reference to closure value extension */
class ClosureValueExt final : public Value {
 public:
  static ClosureValueExt make(ir::Map<ir::Var, Value> env, ir::IRModule mod, ir::GlobalVar gvar);
  MNM_OBJECT_REF(ClosureValueExt, Value, ClosureValueExtObj);
};

}  // namespace value
}  // namespace mnm
