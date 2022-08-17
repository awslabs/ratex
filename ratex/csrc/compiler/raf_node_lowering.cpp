/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./raf_node_lowering.h"

#include <c10/util/BFloat16.h>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
#include "lazy_tensor_core/csrc/ops/all_gather.h"
#include "lazy_tensor_core/csrc/ops/all_reduce.h"
#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"
#include "lazy_tensor_core/csrc/ops/any.h"
#include "lazy_tensor_core/csrc/ops/arg_max.h"
#include "lazy_tensor_core/csrc/ops/arg_min.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy_backward.h"
#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/cholesky.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/cumprod.h"
#include "lazy_tensor_core/csrc/ops/cumsum.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/flip.h"
#include "lazy_tensor_core/csrc/ops/gather.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"
#include "lazy_tensor_core/csrc/ops/hardshrink.h"
#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"
#include "lazy_tensor_core/csrc/ops/index_along_dim.h"
#include "lazy_tensor_core/csrc/ops/index_get.h"
#include "lazy_tensor_core/csrc/ops/index_put.h"
#include "lazy_tensor_core/csrc/ops/index_select.h"
#include "lazy_tensor_core/csrc/ops/kth_value.h"
#include "lazy_tensor_core/csrc/ops/l1_loss.h"
#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"
#include "lazy_tensor_core/csrc/ops/log_base.h"
#include "lazy_tensor_core/csrc/ops/log_softmax.h"
#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/masked_fill.h"
#include "lazy_tensor_core/csrc/ops/masked_scatter.h"
#include "lazy_tensor_core/csrc/ops/max_in_dim.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd.h"
#include "lazy_tensor_core/csrc/ops/mean.h"
#include "lazy_tensor_core/csrc/ops/mse_loss.h"
#include "lazy_tensor_core/csrc/ops/mse_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d_backward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/prod.h"
#include "lazy_tensor_core/csrc/ops/put.h"
#include "lazy_tensor_core/csrc/ops/qr.h"
#include "lazy_tensor_core/csrc/ops/reduce_scatter.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d_backward.h"
#include "lazy_tensor_core/csrc/ops/replication_pad.h"
#include "lazy_tensor_core/csrc/ops/replication_pad_backward.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/scatter.h"
#include "lazy_tensor_core/csrc/ops/scatter_add.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/shrink_backward.h"
#include "lazy_tensor_core/csrc/ops/softmax.h"
#include "lazy_tensor_core/csrc/ops/softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/softshrink.h"
#include "lazy_tensor_core/csrc/ops/split.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/std.h"
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/symeig.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/topk.h"
#include "lazy_tensor_core/csrc/ops/triangular_solve.h"
#include "lazy_tensor_core/csrc/ops/tril.h"
#include "lazy_tensor_core/csrc/ops/triu.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"
#include "lazy_tensor_core/csrc/ops/var.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/ops/embedding.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensor_core/csrc/ops/dropout.h"
#include "ratex/csrc/ops/dropout_backward.h"

#include "ratex/csrc/ops/relay_expr.h"
#include "ratex/csrc/ops/relay_function.h"
#include "ratex/csrc/ops/log_softmax_backward_use_in.h"
#include "ratex/csrc/ops/raf_ops.h"

#include "./raf_lowering_context.h"
#include "./raf_shape_infer.h"
#include "./utils.h"

#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/binding.h"
#include "raf/pass.h"
#include "raf/src/op/regs/schema2value.h"
#include "raf/src/common/shape_utils.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace {

using namespace raf_backend;
using namespace raf;
using namespace raf::tensor;
using namespace raf::ir;
using namespace raf::value;
using namespace raf::binding;
using namespace raf::pass;
using raf::op::regs::schema2value::Bool;
using raf::op::regs::schema2value::Double;
using raf::op::regs::schema2value::Int;
using raf::op::regs::schema2value::String;
using raf::op::regs::schema2value::TupleInt;
using raf::pass::extract_binding::ExtractBinding;

#define DECLARE_OP(name) Var Lower##name(const ir::Node* node)
#define DECLARE_OP2(name) Var Lower##name(const ir::ops::name* node)

class RAFNodeLowering : public NodeLowering {
 public:
  RAFNodeLowering(ir::LoweringContext* loctx) : NodeLowering(loctx) {
  }

  bool Lower(const ir::Node* node) override;

  lazy_tensors::Shape Infer(const ir::Node* node) override;

  raf_backend::RAFLoweringContext* loctx() {
    return static_cast<raf_backend::RAFLoweringContext*>(loctx_);
  }

  Var LowerToRAF(const ir::Node* node);

 private:
  std::tuple<Var, Var> BinaryOpMatchTypes(const ir::Output& a, const ir::Output& b);

  Var LowerBitwise(const ir::Node* node);
  Var LowerAdd(const ir::Node* node);
  Var LowerSub(const ir::Node* node);
  Var LowerDiv(const ir::Node* node);
  Var LowerMul(const ir::Node* node);
  Var LowerDeviceData(const ir::ops::DeviceData* node);
  Var LowerExpand(const ir::ops::Expand* node);
  Var LowerNotSupported(const ir::ops::NotSupported* node);
  template <class NllLossType>
  Var LowerNllLoss(const NllLossType* node);
  template <class NllLossBackwardType>
  Var LowerNllLossBackward(const NllLossBackwardType* node);
  DECLARE_OP(Ne);
  DECLARE_OP(Eq);
  DECLARE_OP(Gt);
  DECLARE_OP(Lt);
  DECLARE_OP(Ceil);
  DECLARE_OP(Abs);
  DECLARE_OP(Pow);
  DECLARE_OP(ReciprocalOp);
  DECLARE_OP(LogicalOr);
  DECLARE_OP2(Constant);
  DECLARE_OP2(Sum);
  DECLARE_OP2(Any);
  DECLARE_OP2(Scalar);
  DECLARE_OP(Relu);
  DECLARE_OP(Sqrt);
  DECLARE_OP(Neg);
  DECLARE_OP(Where);
  DECLARE_OP(Isnan);
  DECLARE_OP2(Permute);
  DECLARE_OP2(MaxPoolNdBackward);
  DECLARE_OP(Mm);
  DECLARE_OP(AddMatMul);
  DECLARE_OP2(ThresholdBackward);
  DECLARE_OP2(MaxPoolNd);
  DECLARE_OP2(LogSoftmax);
  DECLARE_OP2(ConvolutionOverrideable);
  DECLARE_OP2(AdaptiveAvgPool2d);
  DECLARE_OP2(GenericSlice);
  DECLARE_OP2(View);
  DECLARE_OP2(AsStridedViewUpdate);
  DECLARE_OP2(AsStrided);
  DECLARE_OP2(Cast);
  DECLARE_OP2(Dropout);
  DECLARE_OP2(DropoutBackward);
  DECLARE_OP2(LogSoftmaxBackwardUseIn);
  DECLARE_OP2(RelayExpr);
  DECLARE_OP2(RelayFunction);
  DECLARE_OP2(Select);
  DECLARE_OP2(Unselect);
  DECLARE_OP2(ConstantPadNd);
  DECLARE_OP2(Scatter);
  DECLARE_OP2(Cat);
  DECLARE_OP2(Stack);
  DECLARE_OP2(Split);
  DECLARE_OP2(AllReduce);
  DECLARE_OP2(AllGather);
  DECLARE_OP2(ReduceScatter);
  DECLARE_OP2(MaxInDim);
  DECLARE_OP2(ArgMax);
  DECLARE_OP2(Embedding);
  DECLARE_OP(Gelu);
  DECLARE_OP(GeluBackward);
  DECLARE_OP2(Mean);
  DECLARE_OP2(SoftmaxBackward);
  DECLARE_OP2(Softmax);
  lazy_tensors::Shape InferNe(const ir::Node* node);
  lazy_tensors::Shape InferEq(const ir::Node* node);
  lazy_tensors::Shape InferGt(const ir::Node* node);
  lazy_tensors::Shape InferLt(const ir::Node* node);
  lazy_tensors::Shape InferPow(const ir::Node* node);
  lazy_tensors::Shape InferMm(const ir::Node* node);
  lazy_tensors::Shape InferAddMatMul(const ir::Node* node);
  lazy_tensors::Shape InferExpand(const ir::ops::Expand* node);
  lazy_tensors::Shape InferBitwise(const ir::Node* node);
  lazy_tensors::Shape InferLogicalOr(const ir::Node* node);
  lazy_tensors::Shape InferNllLoss(const ir::ops::NllLoss* node);
  lazy_tensors::Shape InferNllLossBackward(const ir::ops::NllLossBackward* node);
  lazy_tensors::Shape InferRelayExpr(const ir::ops::RelayExpr* node);
  lazy_tensors::Shape InferRelayFunction(const ir::ops::RelayFunction* node);
  lazy_tensors::Shape InferAsStridedViewUpdate(const ir::ops::AsStridedViewUpdate* node);
  lazy_tensors::Shape InferDropout(const ir::ops::Dropout* node);
  lazy_tensors::Shape InferDropoutBackward(const ir::ops::DropoutBackward* node);
  lazy_tensors::Shape InferCast(const ir::ops::Cast* node);
  lazy_tensors::Shape InferSum(const ir::ops::Sum* node);
  lazy_tensors::Shape InferAny(const ir::ops::Any* node);
  lazy_tensors::Shape InferConstantPadNd(const ir::ops::ConstantPadNd* node);
  lazy_tensors::Shape InferPermute(const ir::ops::Permute* node);
  lazy_tensors::Shape InferCat(const ir::ops::Cat* node);
  lazy_tensors::Shape InferStack(const ir::ops::Stack* node);
  lazy_tensors::Shape InferSplit(const ir::ops::Split* node);
  lazy_tensors::Shape InferAllReduce(const ir::ops::AllReduce* node);
  lazy_tensors::Shape InferAllGather(const ir::ops::AllGather* node);
  lazy_tensors::Shape InferReduceScatter(const ir::ops::ReduceScatter* node);
  lazy_tensors::Shape InferMaxInDim(const ir::ops::MaxInDim* node);
  lazy_tensors::Shape InferArgMax(const ir::ops::ArgMax* node);
  lazy_tensors::Shape InferConvolutionOverrideable(const ir::ops::ConvolutionOverrideable* node);
  lazy_tensors::Shape InferEmbedding(const ir::ops::Embedding* node);
  lazy_tensors::Shape InferMean(const ir::ops::Mean* node);
};

#undef DECLARE_OP2
#undef DECLARE_OP

bool RAFNodeLowering::Lower(const ir::Node* node) {
  Var ops = LowerToRAF(node);
  if (node->num_outputs() > 1) {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      loctx()->AssignOutputOp(ir::Output(node, i), BindSymbol(TupleGetItem(ops, i)));
    }
  } else {
    loctx()->AssignOutputOp(ir::Output(node, 0), ops);
  }
  return true;
}

#define HANDLE_GENERIC_OP(name, sym) \
  case sym: {                        \
    return Lower##name(node);        \
  }

#define HANDLE_GENERIC_OP2(name, sym)                                       \
  case sym: {                                                               \
    return Lower##name(ir::NodeCast<ir::ops::name>(node, ir::OpKind(sym))); \
  }

Var RAFNodeLowering::LowerToRAF(const ir::Node* node) {
  switch (node->op().op) {
    HANDLE_GENERIC_OP(Add, at::aten::add)
    HANDLE_GENERIC_OP(Sub, at::aten::sub)
    HANDLE_GENERIC_OP(Div, at::aten::div)
    HANDLE_GENERIC_OP(Mul, at::aten::mul)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__and__)
    HANDLE_GENERIC_OP(Relu, at::aten::relu)
    HANDLE_GENERIC_OP(Where, at::aten::where)
    HANDLE_GENERIC_OP(Sqrt, at::aten::sqrt)
    HANDLE_GENERIC_OP(Ceil, at::aten::ceil)
    HANDLE_GENERIC_OP(Neg, at::aten::neg)
    HANDLE_GENERIC_OP(Ne, at::aten::ne)
    HANDLE_GENERIC_OP(Eq, at::aten::eq)
    HANDLE_GENERIC_OP(Gt, at::aten::gt)
    HANDLE_GENERIC_OP(Lt, at::aten::lt)
    HANDLE_GENERIC_OP(Pow, at::aten::pow)
    HANDLE_GENERIC_OP(Abs, at::aten::abs)
    HANDLE_GENERIC_OP(ReciprocalOp, at::aten::reciprocal)
    HANDLE_GENERIC_OP(LogicalOr, at::aten::logical_or)
    HANDLE_GENERIC_OP(Isnan, at::aten::isnan)
    HANDLE_GENERIC_OP2(Permute, at::aten::permute)
    HANDLE_GENERIC_OP2(MaxPoolNdBackward, at::aten::max_pool2d_with_indices_backward)
    HANDLE_GENERIC_OP(Mm, at::aten::mm)
    HANDLE_GENERIC_OP2(NllLoss, at::aten::nll_loss)
    HANDLE_GENERIC_OP2(NllLossBackward, at::aten::nll_loss_backward)
    HANDLE_GENERIC_OP2(Expand, at::aten::expand)
    HANDLE_GENERIC_OP(AddMatMul, at::aten::addmm)
    HANDLE_GENERIC_OP2(ThresholdBackward, at::aten::threshold_backward)
    HANDLE_GENERIC_OP2(MaxPoolNd, at::aten::max_pool2d)
    HANDLE_GENERIC_OP2(LogSoftmax, at::aten::log_softmax)
    HANDLE_GENERIC_OP2(ConvolutionOverrideable, at::aten::convolution_overrideable)
    HANDLE_GENERIC_OP2(View, at::aten::view)
    HANDLE_GENERIC_OP2(AsStrided, at::aten::as_strided)
    HANDLE_GENERIC_OP2(Sum, at::aten::sum)
    HANDLE_GENERIC_OP2(Any, at::aten::any)
    HANDLE_GENERIC_OP2(ConstantPadNd, at::aten::constant_pad_nd)
    HANDLE_GENERIC_OP2(Scatter, at::aten::scatter)
    HANDLE_GENERIC_OP2(Dropout, at::aten::dropout)
    HANDLE_GENERIC_OP2(Cat, at::aten::cat)
    HANDLE_GENERIC_OP2(Stack, at::aten::stack)
    HANDLE_GENERIC_OP2(Split, at::aten::split)
    HANDLE_GENERIC_OP2(MaxInDim, at::aten::max)
    HANDLE_GENERIC_OP2(ArgMax, at::aten::argmax)
    HANDLE_GENERIC_OP2(Embedding, at::aten::embedding)
    HANDLE_GENERIC_OP(Gelu, at::aten::gelu)
    HANDLE_GENERIC_OP(GeluBackward, at::aten::gelu_backward)
    HANDLE_GENERIC_OP2(Mean, at::aten::mean)
    HANDLE_GENERIC_OP2(Softmax, at::aten::softmax)
    HANDLE_GENERIC_OP2(SoftmaxBackward, at::aten::_softmax_backward_data)
    case at::prim::Constant: {
      // TODO(asuhan): rework to remove ambiguity between Scalar and Constant
      // nodes to make dynamic_cast unnecessary.
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      auto constant_node = dynamic_cast<const ir::ops::Constant*>(node);
      LTC_CHECK(constant_node);
      return LowerConstant(constant_node);
    }
    default: {
      if (node->op() == *ir::ops::ltc_cast) {
        return LowerCast(ir::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
      }
      if (node->op() == *ir::ops::ltc_device_data) {
        return LowerDeviceData(ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data));
      }
      if (node->op() == *ir::ops::ltc_generic_slice) {
        return LowerGenericSlice(
            ir::NodeCast<ir::ops::GenericSlice>(node, *ir::ops::ltc_generic_slice));
      }
      if (node->op() == *ir::ops::ltc_as_strided_view_update) {
        return LowerAsStridedViewUpdate(
            ir::NodeCast<ir::ops::AsStridedViewUpdate>(node, *ir::ops::ltc_as_strided_view_update));
      }
      if (node->op() == *ir::ops::ltc_select) {
        return LowerSelect(ir::NodeCast<ir::ops::Select>(node, *ir::ops::ltc_select));
      }
      if (node->op() == *ir::ops::ltc_unselect) {
        return LowerUnselect(ir::NodeCast<ir::ops::Unselect>(node, *ir::ops::ltc_unselect));
      }
      if (node->op() == *ir::ops::raf_relay_expr) {
        return LowerRelayExpr(ir::NodeCast<ir::ops::RelayExpr>(node, *ir::ops::raf_relay_expr));
      }
      if (node->op() == *ir::ops::raf_relay_function) {
        return LowerRelayFunction(
            ir::NodeCast<ir::ops::RelayFunction>(node, *ir::ops::raf_relay_function));
      }
      if (node->op() == *ir::ops::raf_log_softmax_backward_use_in) {
        return LowerLogSoftmaxBackwardUseIn(ir::NodeCast<ir::ops::LogSoftmaxBackwardUseIn>(
            node, *ir::ops::raf_log_softmax_backward_use_in));
      }
      if (node->op() == *ir::ops::ltc_all_gather) {
        return LowerAllGather(ir::NodeCast<ir::ops::AllGather>(node, *ir::ops::ltc_all_gather));
      }
      if (node->op() == *ir::ops::ltc_cross_replica_sum) {
        return LowerAllReduce(
            ir::NodeCast<ir::ops::AllReduce>(node, *ir::ops::ltc_cross_replica_sum));
      }
      if (node->op() == *ir::ops::ltc_reduce_scatter) {
        return LowerReduceScatter(
            ir::NodeCast<ir::ops::ReduceScatter>(node, *ir::ops::ltc_reduce_scatter));
      }
      if (node->op() == *ir::ops::raf_dropout_backward) {
        return LowerDropoutBackward(
            ir::NodeCast<ir::ops::DropoutBackward>(node, *ir::ops::raf_dropout_backward));
      }
    }
  }
  LTC_LOG(FATAL) << "NotImplementedError: " << *node;
  return {};
}

#undef HANDLE_GENERIC_OP2
#undef HANDLE_GENERIC_OP

std::tuple<Var, Var> RAFNodeLowering::BinaryOpMatchTypes(const ir::Output& a, const ir::Output& b) {
  using tvm::runtime::DLDataType2String;
  Var op0 = loctx()->GetOutputOp(a), op1 = loctx()->GetOutputOp(b);
  DType dtype_a = ToRAFDType(a.shape().element_type()),
        dtype_b = ToRAFDType(b.shape().element_type());
  LTC_CHECK_EQ(dtype_a.lanes, dtype_b.lanes);
  // Two cases are supported for binary ops:
  // 1. One of the operands is float, and the other is int. In this case int is casted to float
  // 2. The two operands are of the same type, but of different bits. In this case the
  //    low precision one will be casted to high precision.
  if ((dtype_a.code == DTypeCode::kInt() && dtype_b.code == DTypeCode::kFloat()) ||
      (dtype_a.code == DTypeCode::kBFloat() && dtype_b.code == DTypeCode::kFloat()) ||
      (dtype_a.code == dtype_b.code && dtype_a.bits < dtype_b.bits)) {
    return std::make_tuple(
        BindSymbol(raf::ir::Call(Op::Get("raf.op.cast"),
                                 {op0, MakeConstant(String(DLDataType2String(dtype_b)))})),
        op1);
  } else if ((dtype_a.code == DTypeCode::kFloat() && dtype_b.code == DTypeCode::kInt()) ||
             (dtype_a.code == DTypeCode::kFloat() && dtype_b.code == DTypeCode::kBFloat()) ||
             (dtype_a.code == dtype_b.code && dtype_a.bits > dtype_b.bits)) {
    return std::make_tuple(
        op0, BindSymbol(raf::ir::Call(Op::Get("raf.op.cast"),
                                      {op1, MakeConstant(String(DLDataType2String(dtype_a)))})));
  } else if (dtype_a.code == dtype_b.code && dtype_a.bits == dtype_b.bits) {
    return std::make_tuple(op0, op1);
  } else {
    LTC_LOG(FATAL) << "Not Implemented Error: " << a.shape() << " " << b.shape();
  }
}

bool IsScalar(const ir::Output& x, double val) {
  if (x.node->op().op == at::prim::Constant) {
    const auto* scalar = dynamic_cast<const ir::ops::Scalar*>(x.node);
    LTC_CHECK(scalar);
    return scalar->value().isFloatingPoint() ? scalar->value().toDouble() == val
                                             : static_cast<double>(scalar->value().toLong()) == val;
  }
  return false;
}

ir::Output SimplifyBinaryInputs(const ir::Output& x, const ir::Output& y) {
  if (x.node->op().op == at::aten::expand) {
    lazy_tensors::Shape x_shape = x.shape();
    lazy_tensors::Shape y_shape = y.shape();
    lazy_tensors::Shape in_shape = x.node->operand(0).shape();
    lazy_tensors::Shape in_y_shape = Helpers::GetPromotedBinaryOpShape(in_shape, y_shape);
    lazy_tensors::Shape x_y_shape = Helpers::GetPromotedBinaryOpShape(x_shape, y_shape);
    if (x_y_shape == in_y_shape) {
      return x.node->operand(0);
    }
  }
  return x;
}

Var RAFNodeLowering::LowerAdd(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  if (IsScalar(node->operand(0), 0)) return op1;
  if (IsScalar(node->operand(1), 0)) return op0;
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.add"), {op0, op1, MakeNull(), MakeNull()}));
}

Var RAFNodeLowering::LowerSub(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.subtract"), {op0, op1, MakeNull(), MakeNull()}));
}

Var RAFNodeLowering::LowerDiv(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.divide"), {op0, op1}));
}

Var RAFNodeLowering::LowerMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  if (IsScalar(node->operand(0), 1)) return op1;
  if (IsScalar(node->operand(1), 1)) return op0;
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.multiply"), {op0, op1}));
}

Var RAFNodeLowering::LowerPow(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.power"), {op0, op1}));
}

Var BuildBitwise(const std::vector<Var>& ops, const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  switch (node->op().op) {
    case at::aten::__and__: {
      return BindSymbol(raf::ir::Call(Op::Get("raf.op.logical_and"), {ops[0], ops[1]}));
    }
  }
  LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op();
}

Var RAFNodeLowering::LowerBitwise(const ir::Node* node) {
  Var op0 = loctx()->GetOutputOp(node->operand(0));
  Var op1 = loctx()->GetOutputOp(node->operand(1));
  return BuildBitwise({op0, op1}, node);
}

Var RAFNodeLowering::LowerLogicalOr(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  op0 = BindSymbol(raf::ir::Call(Op::Get("raf.op.cast"), {op0, MakeConstant(String("bool"))}));
  op1 = BindSymbol(raf::ir::Call(Op::Get("raf.op.cast"), {op1, MakeConstant(String("bool"))}));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.logical_or"), {op0, op1}));
}

Var RAFNodeLowering::LowerDeviceData(const ir::ops::DeviceData* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return loctx()->GetParameter(node->data());
}

Var BuildDropout(const std::vector<Var>& ops, const ir::ops::Dropout* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  Expr p = MakeConstant(Double(node->p()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op._contrib_dropout"), {x, p}));
}

Var RAFNodeLowering::LowerDropout(const ir::ops::Dropout* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildDropout({x}, node);
}

Var BuildDropoutBackward(const std::vector<Var>& ops, const ir::ops::DropoutBackward* node) {
  LTC_CHECK_EQ(ops.size(), 3U);
  Var x = ops[0];
  Var mask = ops[1];
  Var reserve_space = ops[2];
  Expr p = MakeConstant(Double(node->p()));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op._contrib_dropout_dx"), {x, mask, reserve_space, p}));
}

Var RAFNodeLowering::LowerDropoutBackward(const ir::ops::DropoutBackward* node) {
  LTC_CHECK_EQ(node->operands().size(), 3U);
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var mask = loctx()->GetOutputOp(node->operand(1));
  Var reserve_space = loctx()->GetOutputOp(node->operand(2));
  return BuildDropoutBackward({x, mask, reserve_space}, node);
}

Var RAFNodeLowering::LowerLogSoftmax(const ir::ops::LogSoftmax* node) {
  Var input = loctx()->GetOutputOp(node->operand(0));
  Expr dim = MakeConstant(ScalarValue::make((int64_t)node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.log_softmax"), {input, dim}));
}

Var RAFNodeLowering::LowerMaxPoolNd(const ir::ops::MaxPoolNd* node) {
  // TODO(@hzfan): return {result, indices}
  Var input = loctx()->GetOutputOp(node->operand(0));
  Expr kernel = MakeConstant(TupleInt(node->kernel_size()));
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt({1}));
  Expr ceil_mode = MakeConstant(Bool(node->ceil_mode()));
  Expr include_pad = MakeConstant(Bool(true));
  Expr layout = MakeConstant(String("NCHW"));
  Var result = BindSymbol(
      raf::ir::Call(Op::Get("raf.op.max_pool2d"),
                    {input, kernel, stride, padding, dilation, ceil_mode, include_pad, layout}));
  Var ret = BindSymbol(raf::ir::Tuple(Array<Expr>({result, raf::ir::Tuple(Array<Expr>({}))})));
  return ret;
}

Var RAFNodeLowering::LowerMaxPoolNdBackward(const ir::ops::MaxPoolNdBackward* node) {
  // TODO(@hzfan): max_pool2d_dx needs y
  Var grad_output = loctx()->GetOutputOp(node->operand(0));
  Var input = loctx()->GetOutputOp(node->operand(1));
  Expr kernel = MakeConstant(TupleInt(node->kernel_size()));
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt({1}));
  Expr ceil_mode = MakeConstant(Bool(node->ceil_mode()));
  Expr include_pad = MakeConstant(Bool(true));
  return BindSymbol(raf::ir::Call(
      Op::Get("raf.op.max_pool2d_dx"),
      {input, grad_output, kernel, stride, padding, dilation, ceil_mode, include_pad}));
}

Var RAFNodeLowering::LowerRelu(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.relu"), {x}));
}

Var RAFNodeLowering::LowerGelu(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.gelu"), {x}));
}

Var RAFNodeLowering::LowerGeluBackward(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var grad = loctx()->GetOutputOp(node->operand(0));
  Var input = loctx()->GetOutputOp(node->operand(1));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.gelu_dx"), {input, MakeNull(), grad}));
}

Var RAFNodeLowering::LowerWhere(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var cond = loctx()->GetOutputOp(node->operand(0));
  Var t_value = loctx()->GetOutputOp(node->operand(1));
  Var f_value = loctx()->GetOutputOp(node->operand(2));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.where"), {cond, t_value, f_value}));
}

Var RAFNodeLowering::LowerSqrt(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.sqrt"), {x}));
}

Var RAFNodeLowering::LowerSoftmax(const ir::ops::Softmax* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr dim = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.softmax"), {x, dim}));
}

Var RAFNodeLowering::LowerSoftmaxBackward(const ir::ops::SoftmaxBackward* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  Var grad = loctx()->GetOutputOp(node->operand(0));
  Var output = loctx()->GetOutputOp(node->operand(1));
  Expr dim = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.softmax_dx"), {output, grad, dim}));
}

Var RAFNodeLowering::LowerIsnan(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.not_equal"), {x, x}));
}

Var BuildSum(const std::vector<Var>& ops, const ir::ops::Sum* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  std::vector<int64_t> dimension_0 = node->dimensions();
  std::vector<int64_t> keep_reduced_dimension_0(
      dimension_0.size(), static_cast<int64_t>(node->keep_reduced_dimensions()));
  Expr dimension = MakeConstant(TupleInt(dimension_0));
  Expr keep_reduced_dimension = MakeConstant(TupleInt(keep_reduced_dimension_0));
  Expr exclude = MakeConstant(Bool(false));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.sum"), {x, dimension, keep_reduced_dimension, exclude}));
}

Var RAFNodeLowering::LowerSum(const ir::ops::Sum* node) {
  // TODO(@hzfan): handle dtype
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildSum({x}, node);
}

Var BuildAny(const std::vector<Var>& ops, const ir::ops::Any* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  Expr dimension = MakeConstant(TupleInt(node->dimensions()));
  Expr keep_reduced_dimension = MakeConstant(Bool(node->keep_reduced_dimensions()));
  Expr exclude = MakeConstant(Bool(false));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.any"), {x, dimension, keep_reduced_dimension, exclude}));
}

Var RAFNodeLowering::LowerAny(const ir::ops::Any* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildAny({x}, node);
}

template <class NllLossType>
Var BuildNllLoss(const std::vector<Var>& ops, const NllLossType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  LTC_CHECK_EQ(ops.size(), 2U);
  Var logits = ops[0];
  Var labels = ops[1];
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.nll_loss"), {labels, logits}));
}

template <class NllLossType>
Var RAFNodeLowering::LowerNllLoss(const NllLossType* node) {
  Var logits = loctx()->GetOutputOp(node->operand(0));
  Var labels = loctx()->GetOutputOp(node->operand(1));
  return BuildNllLoss({logits, labels}, node);
}

template <class NllLossBackwardType>
Var BuildNllLossBackward(const std::vector<Var>& ops, const NllLossBackwardType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  LTC_CHECK_EQ(ops.size(), 3U);
  Var grad_output = ops[0];
  Var logits = ops[1];
  Var labels = ops[2];
  Var normalized_dy =
      BindSymbol(raf::ir::Call(Op::Get("raf.op.reshape"), {grad_output, MakeConstant(TupleInt({})),
                                                           MakeConstant(Bool(false))}));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.nll_loss_dpred"), {normalized_dy, labels, logits}));
}

template <class NllLossBackwardType>
Var RAFNodeLowering::LowerNllLossBackward(const NllLossBackwardType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  Var grad_output = loctx()->GetOutputOp(node->operand(0));
  Var logits = loctx()->GetOutputOp(node->operand(1));
  Var labels = loctx()->GetOutputOp(node->operand(2));
  return BuildNllLossBackward({grad_output, logits, labels}, node);
}

Var BuildExpand(const std::vector<Var>& ops, const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = ops[0];
  std::vector<int64_t> size = node->size();
  lazy_tensors::Shape shape = node->operand(0).node->shape();
  int offset = size.size() - shape.dimensions_size();
  LTC_CHECK_GE(size.size(), shape.dimensions_size());
  for (int i = 0; i < size.size(); ++i) {
    if (i - offset >= 0) {
      LTC_CHECK(shape.dimensions(i - offset) == 1 || size[i] == shape.dimensions(i - offset));
    }
  }
  x = BindSymbol(raf::ir::Call(Op::Get("raf.op.broadcast_to"), {x, MakeConstant(TupleInt(size))}));
  return x;
}

Var RAFNodeLowering::LowerExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildExpand({x}, node);
}

Var BuildAsStridedViewUpdate(const std::vector<Var>& ops,
                             const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  LTC_CHECK_EQ(ops.size(), node->operands().size());
  for (size_t i = 0; i < ops.size(); ++i) {
    ops[i]->checked_type_ = ToRAFType(node->operand(i).shape());
  }
  // TODO(@hzfan): allow transpose
  for (size_t i = 0; i + 1 < node->stride().size(); ++i) {
    LTC_CHECK_GE(node->stride()[i], node->stride()[i + 1]);
  }
  // TODO(@hzfan): allow offset
  LTC_CHECK_EQ(node->storage_offset(), 0);
  // TODO(@hzfan): allow update being a subarray of target
  auto target_tty = Downcast<TensorType>(ops[0]->checked_type());
  auto update_tty = Downcast<TensorType>(ops[1]->checked_type());
  size_t num_dims = target_tty->shape.size();
  LTC_CHECK_EQ(target_tty->dtype, update_tty->dtype);
  LTC_CHECK_EQ(target_tty->shape.size(), update_tty->shape.size());
  LTC_CHECK_EQ(num_dims, node->size().size());
  for (size_t i = 0; i < num_dims; ++i) {
    const auto* target_dim = target_tty->shape[i].as<IntImmNode>();
    const auto* update_dim = update_tty->shape[i].as<IntImmNode>();
    LTC_CHECK(target_dim);
    LTC_CHECK(update_dim);
    LTC_CHECK_EQ(target_dim->value, update_dim->value);
    LTC_CHECK_EQ(target_dim->value, node->size()[i]);
  }
  return ops[1];
}

Var RAFNodeLowering::LowerAsStridedViewUpdate(const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var target = loctx()->GetOutputOp(node->operand(0));
  Var update = loctx()->GetOutputOp(node->operand(1));
  return BuildAsStridedViewUpdate({target, update}, node);
}

Var BuildAsStrided(const std::vector<Var>& ops, const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  LTC_CHECK_EQ(ops.size(), node->operands().size());
  for (size_t i = 0; i < ops.size(); ++i) {
    ops[i]->checked_type_ = ToRAFType(node->operand(i).shape());
  }
  // TODO(@hzfan): allow transpose
  for (size_t i = 0; i + 1 < node->stride().size(); ++i) {
    LTC_CHECK_GE(node->stride()[i], node->stride()[i + 1]);
  }
  // TODO(@hzfan): allow offset
  LTC_CHECK_EQ(node->storage_offset(), 0);
  // TODO(@hzfan): allow slicing an subarray of input
  auto input_tty = Downcast<TensorType>(ops[0]->checked_type());
  size_t num_dims = input_tty->shape.size();
  LTC_CHECK_EQ(num_dims, node->size().size());
  for (size_t i = 0; i < num_dims; ++i) {
    const auto* dim = input_tty->shape[i].as<IntImmNode>();
    LTC_CHECK(dim);
    LTC_CHECK_EQ(dim->value, node->size()[i]);
  }
  return ops[0];
}

Var RAFNodeLowering::LowerAsStrided(const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var input = loctx()->GetOutputOp(node->operand(0));
  return BuildAsStrided({input}, node);
}

Var RAFNodeLowering::LowerRelayExpr(const ir::ops::RelayExpr* node) {
  Var func = loctx()->GetOutputOp(node->operand(0));
  std::vector<Expr> ops;
  for (size_t i = 1; i < node->operands().size(); ++i) {
    ops.push_back(loctx()->GetOutputOp(node->operand(i)));
  }
  return BindSymbol(raf::ir::Call(func, ops));
}

Var RAFNodeLowering::LowerRelayFunction(const ir::ops::RelayFunction* node) {
  return BindSymbol(node->func());
}

Var BuildConvolutionOverrideable(const std::vector<Var>& ops,
                                 const ir::ops::ConvolutionOverrideable* node) {
  Var x = ops[0];
  Var w = ops[1];
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt(node->dilation()));
  Expr groups = MakeConstant(Int(node->groups()));
  Expr layout = MakeConstant(String("NCHW"));
  Expr kernel_layout = MakeConstant(String("OIHW"));
  Expr out_layout = MakeConstant(String("NCHW"));
  bool transposed = node->transposed();
  std::vector<int64_t> output_padding = node->output_padding();
  LTC_CHECK_EQ(transposed, false);
  for (const auto& i : output_padding) {
    LTC_CHECK_EQ(i, 0);
  }
  x = BindSymbol(raf::ir::Call(Op::Get("raf.op.conv2d"), {x, w, stride, padding, dilation, groups,
                                                          layout, kernel_layout, out_layout}));
  if (ops.size() == 3) {
    Var bias = ops[2];
    Expr axis = MakeConstant(Int(1));
    x = BindSymbol(raf::ir::Call(Op::Get("raf.op.bias_add"), {x, bias, axis}));
  }
  return x;
}

Var RAFNodeLowering::LowerConvolutionOverrideable(const ir::ops::ConvolutionOverrideable* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) {
    ops.push_back(loctx()->GetOutputOp(op));
  }
  return BuildConvolutionOverrideable(ops, node);
}

Var BuildLogSoftmaxBackwardUseIn(const std::vector<Var>& ops,
                                 const ir::ops::LogSoftmaxBackwardUseIn* node) {
  LTC_CHECK_EQ(ops.size(), 2U);
  Var dy = ops[0], y = ops[1];

  static auto op_exp = Op::Get("raf.op.exp");
  static auto op_sum = Op::Get("raf.op.sum");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_subtract = Op::Get("raf.op.subtract");

  const Expr& dim = MakeConstant(Int(node->dim()));
  Var exp_y = BindSymbol(raf::ir::Call(op_exp, {y}));
  Expr keep_dims = MakeConstant(ScalarValue::make((int64_t)1));
  Expr exclude = MakeConstant(BoolValue::make(false));
  Var e_1 = BindSymbol(raf::ir::Call(op_sum, {dy, dim, keep_dims, exclude}));
  Var e_2 = BindSymbol(raf::ir::Call(op_multiply, {exp_y, e_1}));
  Var e_3 = BindSymbol(raf::ir::Call(op_subtract, {dy, e_2, MakeNull(), MakeNull()}));
  return e_3;
}

Var RAFNodeLowering::LowerLogSoftmaxBackwardUseIn(const ir::ops::LogSoftmaxBackwardUseIn* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) {
    ops.push_back(loctx()->GetOutputOp(op));
  }
  return BuildLogSoftmaxBackwardUseIn(ops, node);
}

Var BuildPermute(const std::vector<Var>& ops, const ir::ops::Permute* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = ops[0];
  Expr axes = MakeConstant(TupleInt(node->dims()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.transpose"), {x, axes}));
}

Var RAFNodeLowering::LowerPermute(const ir::ops::Permute* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildPermute({x}, node);
}

Var BuildMm(const std::vector<Var>& ops, const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 2) << "Unexpected number of operands";
  Var x = ops[0];
  Var y = ops[1];
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.matmul"), {x, y}));
}

Var RAFNodeLowering::LowerMm(const ir::Node* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var y = loctx()->GetOutputOp(node->operand(1));
  return BuildMm({x, y}, node);
}

Var BuildAddMatMul(const std::vector<Var>& ops, const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 3) << "Unexpected number of operands";
  Var x = ops[0];
  Var y = ops[1];
  Var bias = ops[2];
  Var mm = BindSymbol(raf::ir::Call(Op::Get("raf.op.matmul"), {x, y}));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.add"), {mm, bias, MakeNull(), MakeNull()}));
}

Var RAFNodeLowering::LowerAddMatMul(const ir::Node* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildAddMatMul(ops, node);
}

Var RAFNodeLowering::LowerAdaptiveAvgPool2d(const ir::ops::AdaptiveAvgPool2d* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr shape = MakeConstant(TupleInt(node->output_size()));
  Expr layout = MakeConstant(String("NCHW"));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.adaptive_avg_pool2d"), {x, shape, layout}));
}

Var RAFNodeLowering::LowerGenericSlice(const ir::ops::GenericSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  std::vector<int64_t> limit_indices(node->base_indices().begin(), node->base_indices().end());
  std::transform(limit_indices.begin(), limit_indices.end(), node->sizes().begin(),
                 limit_indices.begin(), std::plus<int64_t>());
  Expr begin = MakeConstant(TupleInt(node->base_indices()));
  Expr end = MakeConstant(TupleInt(limit_indices));
  Expr strides = MakeConstant(TupleInt(std::vector<int64_t>(limit_indices.size(), 1)));
  Expr slice_mode = MakeConstant(String("end"));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.strided_slice"), {x, begin, end, strides, slice_mode}));
}

Var RAFNodeLowering::LowerView(const ir::ops::View* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr shape = MakeConstant(TupleInt(node->output_size()));
  Expr reverse = MakeConstant(Bool(false));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.reshape"), {x, shape, reverse}));
}

Var BuildCast(const std::vector<Var>& ops, const ir::ops::Cast* node) {
  // TODO(@hzfan): handle node->stype() and node->dtype()
  using tvm::runtime::DLDataType2String;
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.cast"),
                    {ops[0], MakeConstant(String(DLDataType2String(ToRAFDType(node->type()))))}));
}

Var RAFNodeLowering::LowerCast(const ir::ops::Cast* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildCast({x}, node);
}

Var BuildMaxInDim(const std::vector<Var>& ops, const ir::ops::MaxInDim* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  Expr axis = MakeConstant(TupleInt({(int64_t)node->dim()}));
  Expr keepdim = MakeConstant(Bool(node->keepdim()));
  Expr exclude = MakeConstant(Bool(false));
  Var max_ret = BindSymbol(raf::ir::Call(Op::Get("raf.op.max"), {x, axis, keepdim, exclude}));
  Var argmax_ret = BindSymbol(raf::ir::Call(Op::Get("raf.op.argmax"), {x, axis, keepdim, exclude}));
  return BindSymbol(raf::ir::Tuple(Array<Expr>({max_ret, argmax_ret})));
}

Var RAFNodeLowering::LowerMaxInDim(const ir::ops::MaxInDim* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildMaxInDim({x}, node);
}

Var BuildArgMax(const std::vector<Var>& ops, const ir::ops::ArgMax* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  Expr axis = MakeConstant(TupleInt({(int64_t)node->dim()}));
  Expr keepdim = MakeConstant(Bool(node->keepdim()));
  Expr exclude = MakeConstant(Bool(false));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.argmax"), {x, axis, keepdim, exclude}));
}

Var RAFNodeLowering::LowerArgMax(const ir::ops::ArgMax* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildArgMax({x}, node);
}

Var BuildEmbedding(const std::vector<Var>& ops, const ir::ops::Embedding* node) {
  LTC_CHECK_EQ(ops.size(), 2U);
  Var x = ops[0];
  Var indices = ops[1];
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.embedding"), {x, indices}));
}

Var RAFNodeLowering::LowerEmbedding(const ir::ops::Embedding* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) {
    ops.push_back(loctx()->GetOutputOp(op));
  }
  return BuildEmbedding(ops, node);
}

Var BuildMean(const std::vector<Var>& ops, const ir::ops::Mean* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  Var x = ops[0];
  Expr axis = MakeConstant(TupleInt(node->dimensions()));
  Expr keep_reduced_dimensions = MakeConstant(Bool(node->keep_reduced_dimensions()));
  Expr exclude = MakeConstant(Bool(false));
  return BindSymbol(
      raf::ir::Call(Op::Get("raf.op.mean"), {x, axis, keep_reduced_dimensions, exclude}));
}

Var RAFNodeLowering::LowerMean(const ir::ops::Mean* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildMean({x}, node);
}

#define DEFINE_COMPARISON_OP(name, op)                                    \
  Var Build##name(const std::vector<Var>& ops, const ir::Node* node) {    \
    LTC_CHECK_EQ(node->num_outputs(), 1);                                 \
    ops[0]->checked_type_ = ToRAFType(node->operand(0).shape());          \
    ops[1]->checked_type_ = ToRAFType(node->operand(1).shape());          \
    Var op0, op1;                                                         \
    std::tie(op0, op1) = PromoteDType(ops[0], ops[1]);                    \
    return BindSymbol(raf::ir::Call(Op::Get("raf.op." #op), {op0, op1})); \
  }                                                                       \
  Var RAFNodeLowering::Lower##name(const ir::Node* node) {                \
    LTC_CHECK_EQ(node->num_outputs(), 1);                                 \
    Var op0 = loctx()->GetOutputOp(node->operand(0));                     \
    Var op1 = loctx()->GetOutputOp(node->operand(1));                     \
    return Build##name({op0, op1}, node);                                 \
  }

#define DEFINE_UNARY_OP(name, op)                                       \
  Var Build##name(const std::vector<Var>& ops, const ir::Node* node) {  \
    LTC_CHECK_EQ(node->num_outputs(), 1);                               \
    return BindSymbol(raf::ir::Call(Op::Get("raf.op." #op), {ops[0]})); \
  }                                                                     \
  Var RAFNodeLowering::Lower##name(const ir::Node* node) {              \
    Var x = loctx()->GetOutputOp(node->operand(0));                     \
    return Build##name({x}, node);                                      \
  }

DEFINE_UNARY_OP(Ceil, ceil)
DEFINE_UNARY_OP(Abs, abs);
DEFINE_UNARY_OP(Neg, negative);
DEFINE_UNARY_OP(ReciprocalOp, reciprocal);
DEFINE_COMPARISON_OP(Ne, not_equal)
DEFINE_COMPARISON_OP(Eq, equal)
DEFINE_COMPARISON_OP(Gt, greater)
DEFINE_COMPARISON_OP(Lt, less)

#undef DEFINE_COMPARISON_OP
#undef DEFINE_UNARY_OP

Var RAFNodeLowering::LowerThresholdBackward(const ir::ops::ThresholdBackward* node) {
  LTC_LOG(FATAL) << "NotImplementedError";
}

Var RAFNodeLowering::LowerConstant(const ir::ops::Constant* node) {
  // TODO(@hzfan): unify LowerConstant for raf/Sunda, raf/CPU, raf/GPU
  // TODO(@hzfan): embed NeuronTensor into constants directly
  LTC_CHECK_EQ(node->num_outputs(), 1);
  auto device = GetCurrentDevice();
  raf::Device raf_device = ToRAFDevice(device.ToString());
  int64_t nbytes = raf::common::shape_utils::BytesCompactTensor(
      Downcast<TensorType>(ToRAFType(node->value().shape())).as<TensorTypeNode>());
  auto buffer = memory_pool::Memory::Alloc(raf_device, nbytes);
  DType dtype;
  std::vector<int64_t> shape;
  std::tie(shape, dtype) = ToRAFShape(node->value().shape());
  PopulateTensorBuffer(node->value().value(), node->value().shape(), buffer->data, nbytes,
                       Device(device));
  auto value = TensorValue::Assemble(raf_device, dtype, shape, {}, buffer->data, buffer);
  return BindSymbol(MakeConstant(value));
}

template <typename T>
TensorValue MakeScalar(T scalar, DType dtype, raf::Device to_dev, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const auto& x : shape) {
    numel = numel * x;
  }
  LTC_CHECK_GT(numel, 0);
  std::vector<T> value(numel, scalar);
  DLTensor tensor;
  tensor.data = value.data();
  // FIXME(multi-node): This can be a problem when we use multi-node environment
  tensor.device = raf::Device(DevType::kCPU(), 0);
  tensor.dtype = dtype;
  tensor.shape = shape.data();
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty(shape, dtype, to_dev);
  array.CopyFrom(&tensor);
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

template <>
TensorValue MakeScalar<bool>(bool scalar, DType dtype, raf::Device to_dev,
                             std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const auto& x : shape) {
    numel = numel * x;
  }
  LTC_CHECK_GT(numel, 0);
  auto cpu_tensor = tvm::runtime::NDArray::Empty(shape, dtype, {DLDeviceType::kDLCPU, 0});
  auto array = reinterpret_cast<bool*>(cpu_tensor->data);
  for (size_t i = 0; i < numel; ++i) {
    array[i] = scalar;
  }
  auto tensor = tvm::runtime::NDArray::Empty(shape, dtype, to_dev);
  tensor.CopyFrom(cpu_tensor);
  return TensorValue::make(Tensor::FromDLPack(tensor.ToDLPack()));
}

Var RAFNodeLowering::LowerScalar(const ir::ops::Scalar* node) {
  using at::operator<<;
  using tvm::runtime::DLDataType2String;
  LTC_CHECK_EQ(node->num_outputs(), 1);
  TensorValue tv;
  auto device = GetCurrentDevice();
  raf::Device raf_device = ToRAFDevice(device.ToString());
  auto raf_dtype = ToRAFDType(node->shape().element_type());
  Span<const int64_t> dimensions = node->shape().dimensions();
// 1. Switch case based on the LTC dtype.
// 2. Get the scalar data from PyTorch and convert to the primitive C type.
// 3. Make a RAF constant expression using the scalar data.
#define ADD_SCALAR_CASE(LTC_TYPE, PT_TYPE, C_TYPE)                                       \
  case lazy_tensors::PrimitiveType::LTC_TYPE: {                                          \
    tv = MakeScalar<C_TYPE>(static_cast<C_TYPE>(node->value().to##PT_TYPE()), raf_dtype, \
                            raf_device,                                                  \
                            std::vector<int64_t>(dimensions.begin(), dimensions.end())); \
    break;                                                                               \
  }

  switch (node->shape().element_type()) {
    ADD_SCALAR_CASE(PRED, Bool, bool);
    ADD_SCALAR_CASE(S8, Char, int8_t);
    ADD_SCALAR_CASE(S16, Short, int16_t);
    ADD_SCALAR_CASE(S32, Int, int32_t);
    ADD_SCALAR_CASE(S64, Long, int64_t);
    ADD_SCALAR_CASE(U8, Char, uint8_t);
    ADD_SCALAR_CASE(U16, Short, uint16_t);
    ADD_SCALAR_CASE(U32, Int, uint32_t);
    ADD_SCALAR_CASE(U64, Long, uint64_t);
    ADD_SCALAR_CASE(F32, Double, float);
    ADD_SCALAR_CASE(F64, Double, double);
    case lazy_tensors::PrimitiveType::F16: {
      tv = MakeScalar<uint16_t>(
          __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(node->value().toDouble()),
          raf_dtype, raf_device, std::vector<int64_t>(dimensions.begin(), dimensions.end()));
      break;
    }
    case lazy_tensors::PrimitiveType::BF16: {
      tv = MakeScalar<uint16_t>(c10::BFloat16(static_cast<float>(node->value().toDouble())).x,
                                raf_dtype, raf_device,
                                std::vector<int64_t>(dimensions.begin(), dimensions.end()));
      break;
    }
    default:
      LTC_LOG(FATAL) << "Unable to lower scalar " << node->value() << " of shape " << node->shape();
  }

#undef DEFINE_SCALAR_CASE

  // FIXME: Somehow BindSymbol(MakeConstant(tv)) doesn't work, so we use a dummy cast op
  // to make it work. Although the dummy cast op will be simplified by SimplifyExpr pass,
  // it is better to avoid dummy ops anyways.
  Var scalar = BindSymbol(
      raf::ir::Call(Op::Get("raf.op.cast"),
                    {MakeConstant(tv), MakeConstant(String(DLDataType2String(raf_dtype)))}));
  return scalar;
}

Var RAFNodeLowering::LowerSelect(const ir::ops::Select* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr begin = MakeConstant(Int(node->start()));
  Expr end = MakeConstant(Int(node->end()));
  Expr stride = MakeConstant(Int(node->stride()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.strided_slice"), {x, begin, end, stride}));
}

Var RAFNodeLowering::LowerUnselect(const ir::ops::Unselect* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var src = loctx()->GetOutputOp(node->operand(1));
  Expr begin = MakeConstant(Int(node->start()));
  Expr end = MakeConstant(Int(node->end()));
  Expr stride = MakeConstant(Int(node->stride()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.strided_set"), {x, src, begin, end, stride}));
}

Var BuildConstantPadNd(const std::vector<Var>& ops, const ir::ops::ConstantPadNd* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = ops[0];
  std::vector<int64_t> pad_vec(node->pad());

  // RAF and PyTorch have different padding axis order. Appending zeros to full axis and reverse it
  while (pad_vec.size() < node->operand(0).shape().dimensions_size() * 2)
    pad_vec.insert(pad_vec.end(), {0, 0});
  std::reverse(pad_vec.begin(), pad_vec.end());
  for (int i = 0; i < pad_vec.size(); i += 2) std::swap(pad_vec[i], pad_vec[i + 1]);

  Expr pad = MakeConstant(TupleInt(pad_vec));
  Expr value = MakeConstant(Double(node->value().toDouble()));
  Expr pad_mode = MakeConstant(String("constant"));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.pad"), {x, pad, value, pad_mode}));
}

Var RAFNodeLowering::LowerConstantPadNd(const ir::ops::ConstantPadNd* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildConstantPadNd({x}, node);
}

Var RAFNodeLowering::LowerScatter(const ir::ops::Scatter* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var idx = loctx()->GetOutputOp(node->operand(1));
  Var src = loctx()->GetOutputOp(node->operand(2));
  Expr axis = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.scatter"), {x, idx, src, axis}));
}

Var BuildCat(const std::vector<Var>& ops, const ir::ops::Cat* node) {
  std::vector<Expr> ops_expr(ops.begin(), ops.end());
  Var x = BindSymbol(raf::ir::Tuple(Array<Expr>(ops_expr)));
  Expr axis = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.concatenate"), {x, axis}));
}

Var RAFNodeLowering::LowerCat(const ir::ops::Cat* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildCat(ops, node);
}

Var BuildStack(const std::vector<Var>& ops, const ir::ops::Stack* node) {
  std::vector<Expr> ops_expr(ops.begin(), ops.end());
  Var x = BindSymbol(raf::ir::Tuple(Array<Expr>(ops_expr)));
  Expr axis = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.stack"), {x, axis}));
}

Var RAFNodeLowering::LowerStack(const ir::ops::Stack* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildStack(ops, node);
}

Var BuildSplit(const std::vector<Var>& ops, const ir::ops::Split* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  std::vector<int64_t> split_sizes = node->split_sizes();
  for (int i = 1; i < split_sizes.size(); i++) {
    split_sizes[i] = split_sizes[i - 1] + split_sizes[i];
  }
  Expr split = MakeConstant(TupleInt(split_sizes));
  Expr axis = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.split"), {x, split, axis}));
}

Var RAFNodeLowering::LowerSplit(const ir::ops::Split* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildSplit(ops, node);
}

Var BuildAllReduce(const std::vector<Var>& ops, const ir::ops::AllReduce* node) {
  using tvm::runtime::DLDataType2String;
  Expr computation;
  if (node->reduce_type() == AllReduceType::kSum) {
    computation = MakeConstant(String("sum"));
  } else if (node->reduce_type() == AllReduceType::kMul) {
    computation = MakeConstant(String("prod"));
  } else if (node->reduce_type() == AllReduceType::kMin) {
    computation = MakeConstant(String("min"));
  } else if (node->reduce_type() == AllReduceType::kMax) {
    computation = MakeConstant(String("max"));
  } else {
    LTC_LOG(FATAL) << "Unsupported Allreduce Type "
                   << lazy_tensors::util::GetEnumValue(node->reduce_type());
  }

  Expr rank_list = MakeConstant(ConvertReplicaGroupsToValue(node->groups()));

  // The last element in the operands is token
  std::vector<Expr> ops_expr(ops.begin(), ops.end() - 1);
  Var allreduce_in = BindSymbol(raf::ir::Tuple(Array<Expr>(ops_expr)));
  Var ret = BindSymbol(
      raf::ir::Call(Op::Get("raf.op._allreduce"), {allreduce_in, computation, rank_list}));
  if (node->scale() != 1.0) {
    DType dtype = ToRAFDType(node->operands()[0].shape().element_type());
    raf::Device dev = ToRAFDevice(GetCurrentDevice().ToString());
    // Take the reverse of the scale to reserve the precision if the data is integer
    double scale_value = 1.0 / node->scale();
    Expr scale = MakeConstantScalar(dtype, scale_value, dev);
    ret = BindSymbol(raf::ir::Call(Op::Get("raf.op.divide"), {ret, scale}));
  }

  // Bind the token back
  return BindSymbol(raf::ir::Tuple(Array<Expr>({ret, ops.back()})));
}

Var RAFNodeLowering::LowerAllReduce(const ir::ops::AllReduce* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildAllReduce(ops, node);
}

Var BuildAllGather(const std::vector<Var>& ops, const ir::ops::AllGather* node) {
  LTC_CHECK_EQ(ops.size(), 2U);
  Var x = ops[0];
  // The last element in the operands is token
  Var token = ops.back();
  Expr dim = MakeConstant(Int(node->dim()));
  Expr rank_list = MakeConstant(ConvertReplicaGroupsToValue(node->groups()));
  Var ret = BindSymbol(raf::ir::Call(Op::Get("raf.op._allgather"), {x, dim, rank_list}));
  return BindSymbol(raf::ir::Tuple(Array<Expr>({ret, token})));
}

Var RAFNodeLowering::LowerAllGather(const ir::ops::AllGather* node) {
  LTC_CHECK_EQ(node->num_outputs(), 2);
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildAllGather(ops, node);
}

Var BuildReduceScatter(const std::vector<Var>& ops, const ir::ops::ReduceScatter* node) {
  LTC_CHECK_EQ(ops.size(), 2U);
  Expr computation;
  if (node->reduce_type() == AllReduceType::kSum) {
    computation = MakeConstant(String("sum"));
  } else if (node->reduce_type() == AllReduceType::kMul) {
    computation = MakeConstant(String("prod"));
  } else if (node->reduce_type() == AllReduceType::kMin) {
    computation = MakeConstant(String("min"));
  } else if (node->reduce_type() == AllReduceType::kMax) {
    computation = MakeConstant(String("max"));
  } else {
    LTC_LOG(FATAL) << "Unsupported Allreduce Type "
                   << lazy_tensors::util::GetEnumValue(node->reduce_type());
  }
  // The last element in the operands is token
  Var token = ops.back();
  Var x = ops[0];
  Expr rank_list = MakeConstant(ConvertReplicaGroupsToValue(node->groups()));
  Var ret =
      BindSymbol(raf::ir::Call(Op::Get("raf.op._reduce_scatter"), {x, computation, rank_list}));
  return BindSymbol(raf::ir::Tuple(Array<Expr>({ret, token})));
}

Var RAFNodeLowering::LowerReduceScatter(const ir::ops::ReduceScatter* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildReduceScatter(ops, node);
}

lazy_tensors::Shape RAFNodeLowering::Infer(const ir::Node* node) {
  const ir::OpKind& kind = node->op();
  switch (kind.op) {
    case at::aten::relu:
    case at::aten::sqrt: {
      return InferUnary(node);
    }
    case at::aten::pow: {
      return InferPow(node);
    }
    case at::aten::ne: {
      return InferNe(node);
    }
    case at::aten::eq: {
      return InferEq(node);
    }
    case at::aten::gt: {
      return InferGt(node);
    }
    case at::aten::expand: {
      return InferExpand(ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    case at::aten::nll_loss: {
      return InferNllLoss(ir::NodeCast<ir::ops::NllLoss>(node, ir::OpKind(at::aten::nll_loss)));
    }
    case at::aten::nll_loss_backward: {
      return InferNllLossBackward(
          ir::NodeCast<ir::ops::NllLossBackward>(node, ir::OpKind(at::aten::nll_loss_backward)));
    }
    case at::aten::sum: {
      return InferSum(ir::NodeCast<ir::ops::Sum>(node, ir::OpKind(at::aten::sum)));
    }
    case at::aten::any: {
      return InferAny(ir::NodeCast<ir::ops::Any>(node, ir::OpKind(at::aten::any)));
    }
    case at::aten::constant_pad_nd: {
      return InferConstantPadNd(
          ir::NodeCast<ir::ops::ConstantPadNd>(node, ir::OpKind(at::aten::constant_pad_nd)));
    }
    case at::aten::__and__:
    case at::aten::__or__:
    case at::aten::__xor__: {
      return InferBitwise(node);
    }
    case at::aten::mm: {
      return InferMm(node);
    }
    case at::aten::addmm: {
      return InferAddMatMul(node);
    }
    case at::aten::permute: {
      return InferPermute(ir::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute)));
    }
    case at::aten::dropout: {
      return InferDropout(ir::NodeCast<ir::ops::Dropout>(node, ir::OpKind(at::aten::dropout)));
    }
    case at::aten::cat: {
      return InferCat(ir::NodeCast<ir::ops::Cat>(node, ir::OpKind(at::aten::cat)));
    }
    case at::aten::stack: {
      return InferStack(ir::NodeCast<ir::ops::Stack>(node, ir::OpKind(at::aten::stack)));
    }
    case at::aten::split: {
      return InferSplit(ir::NodeCast<ir::ops::Split>(node, ir::OpKind(at::aten::split)));
    }
    case at::aten::max: {
      return InferMaxInDim(ir::NodeCast<ir::ops::MaxInDim>(node, ir::OpKind(at::aten::max)));
    }
    case at::aten::argmax: {
      return InferArgMax(ir::NodeCast<ir::ops::ArgMax>(node, ir::OpKind(at::aten::argmax)));
    }
    case at::aten::logical_or: {
      return InferLogicalOr(node);
    }
    case at::aten::convolution_overrideable: {
      return InferConvolutionOverrideable(ir::NodeCast<ir::ops::ConvolutionOverrideable>(
          node, ir::OpKind(at::aten::convolution_overrideable)));
    }
    case at::aten::embedding: {
      return InferEmbedding(
          ir::NodeCast<ir::ops::Embedding>(node, ir::OpKind(at::aten::embedding)));
    }
    case at::aten::mean: {
      return InferMean(ir::NodeCast<ir::ops::Mean>(node, ir::OpKind(at::aten::mean)));
    }
    case at::aten::lt: {
      return InferLt(node);
    }
    default: {
      if (kind == *ir::ops::ltc_generic_slice) {
        return InferGenericSlice(
            ir::NodeCast<ir::ops::GenericSlice>(node, *ir::ops::ltc_generic_slice));
      }
      if (kind == *ir::ops::raf_relay_expr) {
        return InferRelayExpr(ir::NodeCast<ir::ops::RelayExpr>(node, *ir::ops::raf_relay_expr));
      }
      if (kind == *ir::ops::raf_relay_function) {
        return InferRelayFunction(
            ir::NodeCast<ir::ops::RelayFunction>(node, *ir::ops::raf_relay_function));
      }
      if (kind == *ir::ops::ltc_all_gather) {
        return InferAllGather(ir::NodeCast<ir::ops::AllGather>(node, *ir::ops::ltc_all_gather));
      }
      if (kind == *ir::ops::ltc_cross_replica_sum) {
        return InferAllReduce(
            ir::NodeCast<ir::ops::AllReduce>(node, *ir::ops::ltc_cross_replica_sum));
      }
      if (kind == *ir::ops::ltc_reduce_scatter) {
        return InferReduceScatter(
            ir::NodeCast<ir::ops::ReduceScatter>(node, *ir::ops::ltc_reduce_scatter));
      }
      if (kind == *ir::ops::raf_dropout_backward) {
        return InferDropoutBackward(
            ir::NodeCast<ir::ops::DropoutBackward>(node, *ir::ops::raf_dropout_backward));
      }
      LTC_LOG(FATAL) << "Shape inference not supported for operator: " << kind;
    }
  }
}

lazy_tensors::Shape RAFNodeLowering::InferPow(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  auto x_shape = node->operand(0).shape();
  auto y_shape = node->operand(1).shape();
  return Helpers::GetPromotedBinaryOpShape(x_shape, y_shape);
}

lazy_tensors::Shape RAFNodeLowering::InferExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildExpand(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferBitwise(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildBitwise(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferNllLoss(const ir::ops::NllLoss* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildNllLoss(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferNllLossBackward(const ir::ops::NllLossBackward* node) {
  LTC_CHECK_EQ(node->operands().size(), 3U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildNllLossBackward(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferSum(const ir::ops::Sum* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildSum(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferAny(const ir::ops::Any* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildAny(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferConstantPadNd(const ir::ops::ConstantPadNd* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildConstantPadNd(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferDropout(const ir::ops::Dropout* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  ops.push_back(MakeVar("operand", ToRAFType(node->operand(0).shape())));
  Var out = BuildDropout(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferDropoutBackward(const ir::ops::DropoutBackward* node) {
  LTC_CHECK_EQ(node->operands().size(), 3U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildDropoutBackward(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferConvolutionOverrideable(
    const ir::ops::ConvolutionOverrideable* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildConvolutionOverrideable(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferMm(const ir::Node* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildMm(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferAddMatMul(const ir::Node* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildAddMatMul(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferPermute(const ir::ops::Permute* node) {
  std::vector<Var> ops;
  ops.push_back(MakeVar("operand", ToRAFType(node->operand(0).shape())));
  Var out = BuildPermute(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferCat(const ir::ops::Cat* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildCat(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferStack(const ir::ops::Stack* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildStack(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferSplit(const ir::ops::Split* node) {
  std::vector<Var> ops;
  ops.push_back(MakeVar("operand", ToRAFType(node->operand(0).shape())));
  Var out = BuildSplit(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferRelayExpr(const ir::ops::RelayExpr* node) {
  return node->operand(0).shape();
}

lazy_tensors::Shape RAFNodeLowering::InferRelayFunction(const ir::ops::RelayFunction* node) {
  LTC_LOG(FATAL) << "Should not reach here";
}

lazy_tensors::Shape RAFNodeLowering::InferAllReduce(const ir::ops::AllReduce* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildAllReduce(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferAllGather(const ir::ops::AllGather* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildAllGather(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferReduceScatter(const ir::ops::ReduceScatter* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildReduceScatter(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferMaxInDim(const ir::ops::MaxInDim* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildMaxInDim(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferArgMax(const ir::ops::ArgMax* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildArgMax(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferMean(const ir::ops::Mean* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildMean(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape RAFNodeLowering::InferLogicalOr(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  auto x_shape = node->operand(0).shape();
  auto y_shape = node->operand(1).shape();
  return Helpers::GetPromotedBinaryOpShape(x_shape, y_shape);
}

lazy_tensors::Shape RAFNodeLowering::InferEmbedding(const ir::ops::Embedding* node) {
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildEmbedding(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

#define DEFINE_INFER_COMPARISON_OP(name)                                   \
  lazy_tensors::Shape RAFNodeLowering::Infer##name(const ir::Node* node) { \
    std::vector<Var> ops;                                                  \
    for (const auto& x : node->operands()) {                               \
      Var var = MakeVar("operand", ToRAFType(x.shape()));                  \
      ops.push_back(var);                                                  \
    }                                                                      \
    Var out = Build##name(ops, node);                                      \
    Expr body = InferType(ExtractBinding(out, ops));                       \
    return ToLTCShape(body->checked_type());                               \
  }

DEFINE_INFER_COMPARISON_OP(Ne)
DEFINE_INFER_COMPARISON_OP(Eq)
DEFINE_INFER_COMPARISON_OP(Gt)
DEFINE_INFER_COMPARISON_OP(Lt)

#undef DEFINE_INFER_COMPARISON_OP

}  // namespace

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return std::make_unique<compiler::RAFNodeLowering>(loctx);
}

NodeLowering* NodeLowering::Get() {
  static RAFNodeLowering* raf_node_lowering = new RAFNodeLowering(nullptr);
  return raf_node_lowering;
}

namespace raf_backend {

Var LowerNodeToRAF(const ir::Node* node, RAFLoweringContext* loctx) {
  auto node_lowering = NodeLowering::Create(loctx);
  RAFNodeLowering* raf_node_lowering = static_cast<RAFNodeLowering*>(node_lowering.get());
  return raf_node_lowering->LowerToRAF(node);
}

}  // namespace raf_backend

NodeLowering* GetRAFNodeLowering() {
  return NodeLowering::Get();
}

std::unique_ptr<NodeLowering> CreateRAFNodeLowering(ir::LoweringContext* loctx) {
  return NodeLowering::Create(loctx);
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
