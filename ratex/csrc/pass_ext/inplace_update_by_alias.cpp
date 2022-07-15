/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/inplace_update_by_alias.cc
 * \brief Mutate the IR to attach in-place update information according to the given alias map.
 */
#include <unordered_map>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "raf/src/pass/common.h"
#include "raf/src/pass/let_list.h"

namespace raf {

namespace pass {

namespace inplace_update_by_alias {

using namespace raf::ir;
using namespace raf::op;

/*!
 * \brief Mutate the IR to attach in-place update information according to the given alias map.
 * Note that we cannot simply mark may_share because the IR may be transformed to GNF and erase
 * the alias information. Thus, for now we only update the ops with "out" in their arguments.
 */
class InplaceUpdater {
 public:
  InplaceUpdater(const ir::Map<tvm::Integer, tvm::Integer> alias_map) : alias_map_(alias_map) {
  }

  Expr UpdateCall(int out_idx, const Expr& expr, const Var& param) {
    static auto add_op = Op::Get("raf.op.add");
    static auto subtract_op = Op::Get("raf.op.subtract");

    if (!expr->IsInstance<CallNode>()) {
      LOG(WARNING) << "Output." << out_idx
                   << " is not binded to a call node: " << raf::ir::AsText(expr);
      return expr;
    }

    auto call = Downcast<Call>(expr);
    auto op_node = call->op.as<OpNode>();
    if (!op_node) {
      LOG(WARNING) << "Output." << out_idx
                   << " is not binded to an op: " << raf::ir::AsText(call->op);
      return expr;
    }

    auto op = GetRef<Op>(op_node);
    if (op == add_op || op == subtract_op) {
      return Call(op, {call->args[0], call->args[1], param, call->args[3]});
    }

    LOG(WARNING) << "Output." << out_idx
                 << " is not binded to an op with inplace update: " << raf::ir::AsText(op);
    return expr;
  }

  Expr operator()(const Expr& e) {
    auto func = Downcast<Function>(e);
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    std::vector<Var> vars = ell->vars;
    std::vector<Expr> exprs = ell->exprs;
    size_t n = vars.size();
    CHECK_EQ(vars.size(), exprs.size());

    auto out_tuple = exprs[n - 1].as<TupleNode>();

    // Build a map from output var to input param.
    for (auto kv : alias_map_) {
      auto in_idx = kv.first.IntValue();
      auto out_idx = kv.second.IntValue();
      auto out_var_node = (out_tuple != nullptr) ? out_tuple->fields[out_idx].as<VarNode>()
                                                 : vars[n - 1].as<VarNode>();
      CHECK(out_var_node != nullptr)
          << "Output." << out_idx << " is not a var:" << exprs[n - 1]->GetTypeKey();
      auto out_var = GetRef<Var>(out_var_node);
      out_var_to_param_.Set(out_var, func->params[in_idx]);
      out_var_to_idx_.Set(out_var, out_idx);
      DLOG(INFO) << "Alias: out." << out_idx << " and param." << in_idx << " ("
                 << func->params[in_idx]->name_hint() << ")";
    }

    Expr body = LetList::With([&](LetList* ll) {
      for (size_t i = 0; i < n; ++i) {
        auto var = vars[i];
        auto expr = exprs[i];

        // Update the expression for the binded output var that shares an input buffer.
        if (out_var_to_param_.count(var)) {
          expr = UpdateCall(out_var_to_idx_[var].IntValue(), expr, out_var_to_param_[var]);
        }
        ll->Push(var, expr);
      }
      return ell->ret;
    });
    return Function(func->params, body, {}, func->type_params);
  }

 private:
  /*! \brief Mapping from input index to output index that share the same buffer. */
  ir::Map<tvm::Integer, tvm::Integer> alias_map_;
  /*! \brief Mapping from output let-binding var to input param var. */
  ir::Map<ir::Var, ir::Var> out_var_to_param_;
  /*! \brief Mapping from output var to its index. */
  ir::Map<ir::Var, tvm::Integer> out_var_to_idx_;
};

}  // namespace inplace_update_by_alias

Pass InplaceUpdateByAlias(ir::Map<tvm::Integer, tvm::Integer> alias_map) {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        auto entry = ir::Downcast<ir::Function>(mod->Lookup("main"));
        inplace_update_by_alias::InplaceUpdater updater(alias_map);
        ir::BaseFunc updated_entry = ir::Downcast<ir::BaseFunc>(updater(entry));
        ir::IRModule updated_mod = ir::IRModule(mod->functions);
        updated_mod->Add(updated_mod->GetGlobalVar("main"), updated_entry, true);
        return updated_mod;
      },
      1, "InplaceUpdateByAlias", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.InplaceUpdateByAlias").set_body_typed(InplaceUpdateByAlias);

}  // namespace pass
}  // namespace raf
