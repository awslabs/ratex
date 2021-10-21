/*!
 * Copyright (c) 2021 by Contributors
 * \file common.h
 * \brief common utilities
 */
#pragma once
#include <unordered_map>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"
#include "meta/src/pass/common.h"
#include "meta/src/pass/let_list.h"

namespace mnm {
namespace pass {

inline std::pair<mnm::ir::Var, mnm::ir::Expr> FindForwardOutput(const std::vector<Var>& vars,
                                                                const std::vector<Expr>& exprs) {
  size_t n = vars.size();

  // Build a var to expression map.
  mnm::ir::Map<Var, Expr> var_to_expr;
  for (size_t i = 0; i < n; ++i) {
    var_to_expr.Set(vars[i], exprs[i]);
  }

  // Identify the forward output tuple.
  auto fwd_out_var = vars[n - 1];
  if (auto tuple = var_to_expr[fwd_out_var].as<TupleNode>()) {
    if (tuple->fields.size() == 2 &&
        var_to_expr[Downcast<Var>(tuple->fields[1])]->IsInstance<FunctionNode>()) {
      // The second field is a closure, meaning that this is the adjoint
      // closure. In this case, the forward output we are interested in is the
      // first field.
      auto var = tuple->fields[0].as<VarNode>();
      CHECK(var != nullptr);
      return {GetRef<Var>(var), var_to_expr[GetRef<Var>(var)]};
    } else {
      // Otherwise check all fields to make sure there are no closures. This
      // may happen if the adjoint closure is already inlined.
      for (auto field : tuple->fields) {
        if (field->IsInstance<FunctionNode>()) {
          return {Var(), Expr()};
        }
      }
      return {fwd_out_var, GetRef<Tuple>(tuple)};
    }
  }
  return {Var(), Expr()};
}

}  // namespace pass
}  // namespace mnm
