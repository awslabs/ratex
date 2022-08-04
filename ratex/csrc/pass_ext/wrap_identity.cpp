/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/wrap_identity.cc
 * \brief Wrap identity output values with raf.op.copy
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "raf/src/pass/common.h"
#include "raf/src/pass/let_list.h"

namespace raf {

namespace binding {

using namespace raf::ir;
using namespace raf::value;

TensorValue MakeOnes(Device to_dev);

}  // namespace binding

namespace pass {
namespace wrap_identity {

using namespace raf::ir;
using namespace raf::op;

/*! \brief This pass processes the output tuple to make it RAF friendly.
 * Given an output tuple %ret = (%x_0, %x1, ..., %x_n),
 * 1. constants are copied. If %x_i is a constant node, it becomes
 *     let %new_x_i = raf.op.copy(%x_i)
 * 2. input parameters are copied. If %x_i is a input parameter %p_j, it becomes
 *     let %new_x_i = raf.op.copy(%p_j)
 * 3. duplicated fields are copied. If %xi is outputed twice, it becomes
 *     let %new_x_i = raf.op.copy(%x_i)
 *     let %ret = (%x0, %x1, ..., %xi, ..., %new_x_i, ...)
 */
class IdentityWrapper : ExprMutator {
 public:
  IdentityWrapper() {
  }

  Var Copy(const ExprNode* node) {
    const static Op copy = Op::Get("raf.op.copy");
    return ll_->Push(MakeVar("copy" + std::to_string(cnt_++), {}),
                     Call(copy, {GetRef<Expr>(node)}));
  }

  Expr VisitExpr_(const VarNode* node) {
    if (params_.find(GetRef<Var>(node)) != params_.end()) {
      return Copy(node);
    }
    return Downcast<Var>(ExprMutator::VisitExpr_(node));
  }

  Expr VisitExpr_(const RelayConstantNode* node) {
    return Copy(node);
  }

  Expr operator()(const Expr& e) {
    auto func = Downcast<Function>(e);
    if (!func->body.as<LetNode>()) {
      // Function is not in ANF
      if (func->body.as<RelayConstantNode>()) {
        Expr body = LetList::With([&](LetList* ll) {
          const static Op copy = Op::Get("raf.op.copy");
          Var v = MakeVar("copy" + std::to_string(cnt_++), {});
          return ll->Push(v, Call(copy, {func->body}));
        });
        return Function(func->params, body, func->ret_type, func->type_params);
      } else {
        LOG(FATAL) << "Unsupported type " << func->body->GetTypeKey();
      }
    }
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    std::vector<Var> vars = ell->vars;
    std::vector<Expr> exprs = ell->exprs;
    size_t n = vars.size();
    CHECK_EQ(vars.size(), exprs.size());
    for (const auto& param : func->params) {
      params_.insert(param);
    }
    Expr body = LetList::With([&](LetList* ll) {
      ll_ = ll;
      for (size_t i = 0; i + 1 < n; ++i) {
        ll->Push(vars[i], exprs[i]);
      }
      Expr ret = exprs[n - 1];
      // ret can be of two nodes: TupleNode or CallNode
      // For TupleNode, each of its fields cannot be constants or input parameters
      if (ret.as<TupleNode>()) {
        ret = VisitExpr(ret);
      }
      // Deduplicate
      // We cannot use ExprMutator for deduplicate, because ExprMutator is cached
      // The second time it sees a variable, the cache will be hit and thus gives
      // the same result as the first use of this variable.
      if (const auto* tup = ret.as<TupleNode>()) {
        Array<Expr> arr;
        for (const auto& i : tup->fields) {
          auto var = Downcast<Var>(i);
          if (outputs_.find(var) != outputs_.end()) {
            arr.push_back(Copy(var.as<VarNode>()));
          } else {
            outputs_.insert(var);
            arr.push_back(var);
          }
        }
        ret = Tuple(arr);
      }
      ll->Push(vars[n - 1], ret);
      return ell->ret;
    });
    return Function(func->params, body, func->ret_type, func->type_params);
  }

 private:
  LetList* ll_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> params_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> outputs_;
  int cnt_{0};
};

}  // namespace wrap_identity

Pass WrapIdentity() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(wrap_identity::IdentityWrapper()(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "WrapIdentity", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.WrapIdentity").set_body_typed(WrapIdentity);

}  // namespace pass
}  // namespace raf
