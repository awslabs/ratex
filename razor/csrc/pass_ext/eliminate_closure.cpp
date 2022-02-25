/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/eliminate_closure.cc
 * \brief Eliminate closure value in function return
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"
#include "mnm/value.h"
#include "meta/src/pass/common.h"
#include "meta/src/pass/let_list.h"

namespace mnm {

namespace binding {

using namespace mnm::ir;
using namespace mnm::value;

TensorValue MakeOnes(Device to_dev);

}  // namespace binding

namespace pass {
namespace eliminate_closure {

using namespace mnm::ir;
using namespace mnm::op;

class ClosureEliminator {
 public:
  ClosureEliminator() {
  }

  Expr EliminateClosure(Var x, Type ty = Type()) {
    if (!ty.defined()) {
      ty = x->checked_type();
    }
    if (ty.as<TupleTypeNode>()) {
      Array<Type> fields = ty.as<TupleTypeNode>()->fields;
      Array<Expr> updated_fields;
      for (size_t i = 0; i < fields.size(); ++i) {
        updated_fields.push_back(EliminateClosure(ll_->Push(TupleGetItem(x, i)), fields[i]));
      }
      return ll_->Push(Tuple(updated_fields));
    } else if (ty.as<TensorTypeNode>()) {
      return x;
    } else if (ty.as<FuncTypeNode>()) {
      const static auto& ones = Op::Get("mnm.op.ones");
      return ll_->Push(Call(ones, {MakeConstant(mnm::value::TupleValue::make({})),
                                   MakeConstant(mnm::value::StringValue::make("float32")),
                                   MakeConstant(mnm::value::StringValue::make("cpu"))}));
    }
    LOG(FATAL) << "Unsupported type: " << ty;
  }

  Expr operator()(const Expr& e) {
    auto func = Downcast<Function>(e);
    if (!func->body.as<LetNode>()) {
      // Function is not in ANF
      return func;
    }
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    std::vector<Var> vars = ell->vars;
    std::vector<Expr> exprs = ell->exprs;
    size_t n = vars.size();
    CHECK_EQ(vars.size(), exprs.size());
    Expr body = LetList::With([&](LetList* ll) {
      ll_ = ll;
      for (size_t i = 0; i < n; ++i) {
        ll->Push(vars[i], exprs[i]);
      }
      Var ret = ell->ret;
      return EliminateClosure(ret);
    });
    return Function(func->params, body, {}, func->type_params);
  }

 private:
  LetList* ll_;
};

}  // namespace eliminate_closure

Pass EliminateClosure() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        auto entry = ir::Downcast<ir::Function>(mod->Lookup("main"));
        eliminate_closure::ClosureEliminator ce;
        ir::BaseFunc updated_entry = ir::Downcast<ir::BaseFunc>(ce(entry));
        ir::IRModule updated_mod = ir::IRModule(mod->functions);
        updated_mod->Add(updated_mod->GetGlobalVar("main"), updated_entry, true);
        return updated_mod;
      },
      1, "EliminateClosure", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.EliminateClosure").set_body_typed(EliminateClosure);

}  // namespace pass
}  // namespace mnm
