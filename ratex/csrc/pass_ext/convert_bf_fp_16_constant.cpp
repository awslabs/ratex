/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/convert_bf_fp_16_constant.cc
 * \brief Mutate the IR as describled below:
 * (1) cast all the constant float32 tensors to constant bf16/fp16 tensors;
 * (2) if a cast call's target dtype is float32, we change the target dtype to bf16/fp16
 */
#include "raf/op.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace convert_bf_fp_16_constant {

using namespace raf::value;

class BfFp16ConstantConverter : public ExprMutator {
 public:
  BfFp16ConstantConverter(tvm::String bf_fp_16_dtype) : bf_fp_16_dtype_(bf_fp_16_dtype.c_str()) {
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* let_node) {
      Expr value = this->Mutate(let_node->value);

      if (this->is_constant_float32_tensor(value)) {
        // map float32 constant tenosr to a new bf16/fp16 constant tensor
        this->memo_[let_node->var] =
            MakeVar(let_node->var->name_hint() + "_" + bf_fp_16_dtype_, {});
      } else {
        this->Mutate(let_node->var);
      }
    };
    auto post_visit = [this](const LetNode* let_node) {
      Expr expr = GetRef<Expr>(let_node);

      Var var = Downcast<Var>(this->Mutate(let_node->var));
      Expr value = this->Mutate(let_node->value);
      Expr body = this->Mutate(let_node->body);

      auto const_val = value.as<ConstantNode>();
      if (this->is_constant_float32_tensor(value)) {
        assert(!var.same_as(let_node->var));
        // cast the float32 constant tensor to the new bf16/fp16 constant tensor
        auto cast_call =
            Call(Op::Get("raf.op.cast"),
                 {let_node->var, MakeConstant(StringValue::make(bf_fp_16_dtype_))}, {});
        this->memo_[expr] = Let(let_node->var, value, Let(var, cast_call, body));
      } else {
        if (var.same_as(let_node->var) && value.same_as(let_node->value) &&
            body.same_as(let_node->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr res = ExprMutator::VisitExpr_(call_node);

    // if the call is a cast whose target dtype is float32, we change the target dtype to bf16/fp16
    call_node = res.as<CallNode>();
    auto op = call_node->op.as<OpNode>();
    if (op == nullptr || op->name != "raf.op.cast") return res;
    assert(call_node->args.size() == 2 && call_node->args[1].as<ConstantNode>() &&
           call_node->args[1].as<ConstantNode>()->value.as<StringValueObj>());
    auto cast_dest_type = call_node->args[1].as<ConstantNode>()->value.as<StringValueObj>()->value;
    if (cast_dest_type == "float32") {
      return Call(call_node->op,
                  {call_node->args[0], MakeConstant(StringValue::make(bf_fp_16_dtype_))}, {},
                  call_node->type_args);
    } else {
      return res;
    }
  }

 private:
  // bf_fp_16_dtype indicates the target dtype (bf16 or fp16)
  const char* bf_fp_16_dtype_;

  // helper function to determine whether an Expr is a constant float32 tensor
  bool is_constant_float32_tensor(Expr value) const {
    auto const_val = value.as<ConstantNode>();
    if (const_val && const_val->IsTensor()) {
      auto dtype = tvm::runtime::DataType(Downcast<TensorValue>(const_val->value)->tensor->dtype);
      return dtype.is_float() && dtype.bits() == 32;
    }
    return false;
  }
};

}  // namespace convert_bf_fp_16_constant

Pass ConvertBfFp16Constant(tvm::String bf_fp_16_dtype) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(
        convert_bf_fp_16_constant::BfFp16ConstantConverter(bf_fp_16_dtype).Mutate(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "ConvertBfFp16Constant", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ConvertBfFp16Constant").set_body_typed(ConvertBfFp16Constant);

}  // namespace pass
}  // namespace raf
