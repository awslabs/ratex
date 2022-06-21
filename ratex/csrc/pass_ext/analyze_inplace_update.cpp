/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/analyze_inplace_update.cc
 * \brief Analyze the IR to get a mapping from output tuple indices to input
 * argument indices that share the same memory (inplace update).
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

using PackedVarIdxMap = Map<Integer, Integer>;

namespace analyze_inplace_update {

using namespace raf::ir;
using namespace raf::op;

/*!
 * \brief Find the output tuple index that corresponds to the input argument. Note that this pass
 * has to be applied after CanonicalizeParamsForRATEX; otherwise the forward output tuple may be
 * in a nested tuple (%fwd_out_tuple, %adjoint_closure), which won't be correctly analyzed.
 */
class InplaceUpdateAnalyzer {
 public:
  InplaceUpdateAnalyzer() {
  }

  PackedVarIdxMap operator()(const Expr& e) {
    auto func = Downcast<Function>(e);
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    std::vector<Var> vars = ell->vars;
    std::vector<Expr> exprs = ell->exprs;
    size_t n = vars.size();
    CHECK_EQ(vars.size(), exprs.size());

    // Build a var to argument index map.
    Map<Var, Integer> arg_to_idx;
    for (size_t i = 0; i < func->params.size(); ++i) {
      arg_to_idx.Set(func->params[i], i);
    }

    auto out_tuple = exprs[n - 1].as<TupleNode>();
    if (out_tuple == nullptr) {
      return inplace_var_map_;
    }

    for (size_t i = 0; i < out_tuple->fields.size(); ++i) {
      CHECK(out_tuple->fields[i].as<VarNode>())
          << "Expected to return a Var, but got " << out_tuple->fields[i]->GetTypeKey();
      auto var = Downcast<Var>(out_tuple->fields[i]);
      const auto* extended_var = static_cast<const ExtendedVarNode*>(var.operator->());
      if (extended_var && extended_var->may_share.defined()) {
        auto arg_var = extended_var->may_share;
        CHECK_EQ(arg_to_idx.count(arg_var), 1U) << "Output Var inplace updates a non-input tensor";
        inplace_var_map_.Set(i, arg_to_idx[arg_var]);
      }
    }
    return inplace_var_map_;
  }

 private:
  PackedVarIdxMap inplace_var_map_;
};

}  // namespace analyze_inplace_update

PackedVarIdxMap InplaceUpdateAnalysis(const IRModule& mod) {
  auto entry = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(entry));
  return analyze_inplace_update::InplaceUpdateAnalyzer()(func);
}

RAF_REGISTER_GLOBAL("raf.pass_.InplaceUpdateAnalysis").set_body_typed(InplaceUpdateAnalysis);

}  // namespace pass
}  // namespace raf
