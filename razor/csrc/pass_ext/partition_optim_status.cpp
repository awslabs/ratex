/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * Copyright (c) 2021 by Contributors
 * \file partition_optim_status.cc
 * \brief This pass will be used to apply ZeRO1 optimization, which partitions the model states into
 * N ranks, and allgather all the partial values to update the weight. Note that this pass only does
 * the last step, which finds the IR patterns and replaces with allgather ops. The actual optimizer
 * states buffer partition is done as we create the optimizer. The pass looks for this pattern to
 * replace: fn (...) {
 *  ...
 *  let %output_0 = raf.op.scatter( ... )
 *  ...
 *  let %x_3 = (%output_0, %output_1, %output_2);
 *  %x_3
 * }
 *
 * After subsititution:
 * fn (...) {
 *  ...
 *  let %output_0 = raf.op._allgather(%x_1, 0);
 *  ...
 *  let %x_3 = (%output_0, %output_1, %output_2);
 *  %x_3
 * }
 *
 * FIXME: This pass will be removed after we support the tracing of real torch distributed ops
 * https://github.com/raf-project/torch_raf/issues/42
 */

#include "raf/pass.h"
#include "raf/dist_context.h"

#include "raf/src/pass/common.h"
#include "raf/src/pass/let_list.h"
#include "raf/src/common/shape_utils.h"

namespace raf {
namespace pass {
namespace partition_optim_status {

using namespace raf::common::shape_utils;
using raf::distributed::DistContext;

class OptimStatusPartitioner : public ExprMutator {
 public:
  OptimStatusPartitioner(const Function& func) : func_(func) {
    // Build the var to expr map for the ANF.
    auto ell = ExplicitLetList::make(func->body);

    if (auto ret = ell->exprs.back().as<TupleNode>()) {
      for (const auto& field : ret->fields)
        if (field->IsInstance<VarNode>()) outputs_.Set(field, Expr());
    } else if (auto ret = ell->exprs.back().as<CallNode>()) {
      outputs_.Set(ell->exprs.back(), Expr());
    } else {
      LOG_FATAL << "Unexpected output type " << ell->exprs.back()->GetTypeKey();
    }
  }

  /*! \brief Partition the parameters according to the parameter group. */
  Function Run() {
    if (outputs_.empty()) {
      return func_;
    }

    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto curr_expr = VisitExpr(node->value);

      if (IsAllGatherCandidate(curr_var, curr_expr)) {
        auto scatter_call = Downcast<Call>(curr_expr);
        // The 3nd arg of scatter (src) is the data we care about
        auto src_arg = scatter_call->args[2];

        // Replace the binding to allgather
        static const Op& allgather_op = Op::Get("raf.op._allgather");
        auto new_var =
            scope->Push(Call(allgather_op, {src_arg, MakeConstant(ScalarValue::make(int64_t(0)))}));

        // If the parameter shape is not divisible by the number of ranks, we need an extra slice
        auto dctx = DistContext::Global();
        int64_t allgather_first_dim = GetDimSize(src_arg, 0) * dctx->size;
        int64_t actual_first_dim = GetDimSize(scatter_call->args[0], 0);
        CHECK(allgather_first_dim >= actual_first_dim)
            << "First dim of the all gather result must be greater or equal to the first dim of "
               "actual weight, but got "
            << allgather_first_dim << " vs. " << actual_first_dim;
        if (allgather_first_dim > actual_first_dim) {
          static const Op& strided_slice_op = Op::Get("raf.op.strided_slice");
          new_var =
              scope->Push(Call(strided_slice_op, {new_var, MakeConstant(ScalarValue::make(0)),
                                                  MakeConstant(ScalarValue::make(actual_first_dim)),
                                                  MakeConstant(ScalarValue::make(1))}));
        }
        scope->Push(curr_var, new_var);
      } else {
        scope->Push(curr_var, curr_expr);
      }
      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }

 private:
  /*! \brief Check whether a given expression is a call expression with scatter. */
  inline bool IsScatterCall(const Expr& expr) {
    static const Op& scatter_op = Op::Get("raf.op.scatter");
    if (!expr->IsInstance<CallNode>()) {
      return false;
    }
    auto call = Downcast<Call>(expr);
    if (auto node = call->op.as<OpNode>()) {
      return GetRef<Op>(node) == scatter_op;
    }
    return false;
  }

  /*! \brief Return true if it is a candidate for the substitution. */
  bool IsAllGatherCandidate(const Var& var, const Expr& expr) {
    // If the output is a tuple, then all vars in tuple are saved in weight_updates_. If there is
    // only a single output, the back of ExplicitLetList is a Call and the expr is saved in
    // weight_updates_. When we visit the LetNode again, if the graph has single output, then in
    // weight_updates_ we have the expr instead of var
    if (outputs_.size() == 1) return (outputs_.count(expr) > 0) && IsScatterCall(expr);
    return (outputs_.count(var) > 0) && IsScatterCall(expr);
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;

  /*! \brief The target function. */
  Function func_;

  /*! \brief Record the Expr that possibly generate a new weight value. */
  Map<Expr, Expr> outputs_;
};

}  // namespace partition_optim_status

Pass PartitionOptimStatus() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return partition_optim_status::OptimStatusPartitioner(f).Run();
  };
  auto partition_optim_status = CreateRAFFunctionPass(pass_func, 0, "PartitionOptimStatusFunc", {});
  return RAFSequential({partition_optim_status, EraseType(), DeadCodeElimination()},
                       "PartitionOptimStatus");
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionOptimStatus").set_body_typed(PartitionOptimStatus);

}  // namespace pass
}  // namespace raf