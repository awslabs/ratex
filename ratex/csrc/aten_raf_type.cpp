/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/LazyNativeFunctions.h"

#include <ATen/Context.h>
#include <ATen/Operators.h>
#include <ATen/native/Activation.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <c10/util/OptionalArrayRef.h>

#include <mutex>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/debug_util.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/computation_client/nnc_computation_client.h"

#include "ratex/csrc/ops/log_softmax_backward_use_in.h"
#include "ratex/csrc/aten_autograd_ops.h"
#include "ratex/csrc/aten_cpu_fallback.h"
#include "ratex/csrc/version.h"
#include "ratex/csrc/aten_raf_bridge.h"
#include "ratex/csrc/utils/debug.h"
#include "ratex/csrc/utils/ratex_logging.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't exist in AtenRAFType,
//   call at::native::func instead.
//   E.g. don't call tensor.is_floating_point() or
//   at::is_floating_point(tensor), use at::native::is_floating_point(tensor).

namespace torch_lazy_tensors {

bool IsSupportedAdaptiveAvgPool(absl::Span<const int64_t> input_size,
                                absl::Span<const int64_t> output_size, int pool_dim) {
  int64_t rank = input_size.size();
  LTC_CHECK_EQ(output_size.size(), pool_dim);
  for (int spatial_dim = 0; spatial_dim < pool_dim; ++spatial_dim) {
    if (input_size[rank - pool_dim + spatial_dim] % output_size[spatial_dim] != 0) {
      return false;
    }
  }
  return true;
}

bool IsNonTrivialDilation(at::IntArrayRef dilation) {
  return std::any_of(dilation.begin(), dilation.end(),
                     [](const int64_t dim_dilation) { return dim_dilation != 1; });
}

namespace {

Device GetLtcDeviceOrCurrent(const c10::optional<c10::Device>& device) {
  auto ratex_device_opt = bridge::GetLtcDevice(device);
  return ratex_device_opt ? *ratex_device_opt : GetCurrentDevice();
}

at::ScalarType GetScalarTypeOrFloat(c10::optional<at::ScalarType> scalar_type) {
  return scalar_type ? *scalar_type : at::ScalarType::Float;
}

bool IsOperationOnType(const c10::optional<at::ScalarType>& opt_dtype, at::ScalarType tensor_type,
                       at::ScalarType type) {
  if (opt_dtype && *opt_dtype == type) {
    return true;
  }
  return tensor_type == type;
}

void CheckSubOperandTypes(at::ScalarType type1, at::ScalarType type2) {
  LTC_CHECK(type1 != at::kBool || type2 != at::kBool)
      << "Subtraction, the `-` operator, with two bool tensors is not "
         "supported. Use the `^` or `logical_xor()` operator instead.";
  LTC_CHECK(type1 != at::kBool && type2 != at::kBool)
      << "Subtraction, the `-` operator, with a bool tensor is not "
         "supported. If you are trying to invert a mask, use the `~` or "
         "`logical_not()` operator instead.";
}

c10::optional<at::ScalarType> PromoteIntegralType(at::ScalarType src_dtype,
                                                  const c10::optional<at::ScalarType>& opt_dtype) {
  if (opt_dtype.has_value()) {
    return opt_dtype.value();
  }
  if (at::isIntegralType(src_dtype, /*includeBool=*/true)) {
    return at::kLong;
  }
  return opt_dtype;
}

bool IsTypeWithLargerRangeThanLong(at::ScalarType dtype) {
  return dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Float ||
         dtype == at::ScalarType::Double;
}

// Return the upper limit for a given type. For floating point typesreturn
// 2^mantissa to ensure that every value is representable.
int64_t GetIntegerUpperLimitForType(at::ScalarType dtype) {
  lazy_tensors::PrimitiveType ltc_type = TensorTypeToLtcType(dtype);
  switch (ltc_type) {
    case lazy_tensors::PrimitiveType::F16:
      return static_cast<int64_t>(1) << std::numeric_limits<at::Half>::digits;
    case lazy_tensors::PrimitiveType::BF16:
      return static_cast<int64_t>(1) << std::numeric_limits<at::BFloat16>::digits;
    case lazy_tensors::PrimitiveType::F32:
      return static_cast<int64_t>(1) << std::numeric_limits<float>::digits;
    case lazy_tensors::PrimitiveType::F64:
      return static_cast<int64_t>(1) << std::numeric_limits<double>::digits;
    default:
      return Helpers::MinMaxValues(ltc_type).max.toLong();
  }
}

void CheckRangeValues(at::ScalarType dtype, int64_t from, int64_t to) {
  Helpers::MinMax min_max;
  // Bound the min_max by int64 since types of "from" and "to" are int64.
  if (IsTypeWithLargerRangeThanLong(dtype)) {
    min_max = Helpers::MinMaxValues(lazy_tensors::PrimitiveType::S64);
  } else {
    min_max = Helpers::MinMaxValues(TensorTypeToLtcType(dtype));
  }
  LTC_CHECK_GE(from, min_max.min.toLong());
  LTC_CHECK_LE(from, min_max.max.toLong());
  LTC_CHECK_GE(to, min_max.min.toLong());
  LTC_CHECK_LE(to, min_max.max.toLong());
}

std::pair<LazyTensor, LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  LazyTensor self_tensor;
  LazyTensor other_tensor;
  auto self_xtensor = bridge::raf_backend::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::raf_backend::GetLtcTensor(other);
    self_tensor = bridge::GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

// The input is in format of {N, C, H, W} and the output will be {H, W}.
std::vector<int64_t> GetOutputSizeWithScale(
    absl::Span<const int64_t> input_size, const c10::optional<at::ArrayRef<double>>& scale_factors,
    const at::OptionalIntArrayRef& output_size) {
  if (!output_size) {
    LTC_CHECK(scale_factors);
    LTC_CHECK_EQ(scale_factors->size(), 2);
    // Calculate the output size from input_shape and scale_factors
    LTC_CHECK_EQ(input_size.size(), 4);
    int64_t output_h = input_size[2] * (*scale_factors)[0];
    int64_t output_w = input_size[3] * (*scale_factors)[1];
    return {output_h, output_w};
  }
  LTC_CHECK(!scale_factors);
  return lazy_tensors::util::ToVector<int64_t>(*output_size);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other, const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<LazyTensor, LazyTensor> operands = GetBinaryOperands(self, UnwrapNumber(other, dtype));
  LazyTensor result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other, const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor result = bin_op(self_tensor, other, dtype);
  return bridge::AtenFromLtcTensor(result);
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Scalar& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

}  // namespace

at::Tensor& LazyNativeFunctions::__ilshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& LazyNativeFunctions::__ilshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, bridge::raf_backend::GetLtcTensor(other));
  return self;
}

at::Tensor& LazyNativeFunctions::__irshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& LazyNativeFunctions::__irshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, bridge::raf_backend::GetLtcTensor(other));
  return self;
}

at::Tensor LazyNativeFunctions::__lshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor LazyNativeFunctions::__lshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor LazyNativeFunctions::__rshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor LazyNativeFunctions::__rshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor LazyNativeFunctions::_adaptive_avg_pool3d(const at::Tensor& self,
                                                     at::IntArrayRef output_size) {
  LTC_FN_COUNTER("raf::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size_list,
                                  /*pool_dim=*/3)) {
    return FALLBACK_ATEN_OP(_adaptive_avg_pool3d, self, output_size);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::adaptive_avg_pool3d(bridge::raf_backend::GetLtcTensor(self), output_size_list));
}

at::Tensor LazyNativeFunctions::_adaptive_avg_pool3d_backward(const at::Tensor& grad_output,
                                                              const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 3), grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/3)) {
    return FALLBACK_ATEN_OP(_adaptive_avg_pool3d_backward, grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::_adaptive_avg_pool2d(const at::Tensor& self,
                                                     at::IntArrayRef output_size) {
  LTC_FN_COUNTER("raf::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size_list,
                                  /*pool_dim=*/2)) {
    return FALLBACK_ATEN_OP(_adaptive_avg_pool2d, self, output_size);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::_adaptive_avg_pool2d(bridge::raf_backend::GetLtcTensor(self), output_size_list));
}

at::Tensor LazyNativeFunctions::_adaptive_avg_pool2d_backward(const at::Tensor& grad_output,
                                                              const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 2), grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/2)) {
    return FALLBACK_ATEN_OP(_adaptive_avg_pool2d_backward, grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self)));
}

void LazyNativeFunctions::_amp_foreach_non_finite_check_and_unscale_(at::TensorList self,
                                                                     at::Tensor& found_inf,
                                                                     const at::Tensor& inv_scale) {
  LTC_FN_COUNTER("raf::");
  LazyTensor found_inf_tensor = bridge::raf_backend::GetLtcTensor(found_inf);
  DeviceType hw_type = found_inf_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with CPU or GPU";
  LazyTensor::_amp_foreach_non_finite_check_and_unscale_(
      bridge::raf_backend::GetLtcTensors(self), found_inf_tensor,
      bridge::raf_backend::GetLtcTensor(inv_scale));
}

at::Tensor& LazyNativeFunctions::_amp_update_scale_(
    at::Tensor& current_scale, at::Tensor& growth_tracker, const at::Tensor& found_inf,
    double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  LTC_FN_COUNTER("raf::");
  LazyTensor growth_tracker_tensor = bridge::raf_backend::GetLtcTensor(growth_tracker);
  LazyTensor current_scale_tensor = bridge::raf_backend::GetLtcTensor(current_scale);
  DeviceType hw_type = growth_tracker_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with CPU or GPU";
  LazyTensor::_amp_update_scale_(growth_tracker_tensor, current_scale_tensor,
                                 bridge::raf_backend::GetLtcTensor(found_inf), scale_growth_factor,
                                 scale_backoff_factor, growth_interval);
  return current_scale;
}

at::Tensor LazyNativeFunctions::_copy_from(const at::Tensor& self, const at::Tensor& dst,
                                           bool non_blocking) {
  LTC_FN_COUNTER("raf::");
  auto dst_tensor = bridge::raf_backend::TryGetLtcTensor(dst);
  auto self_tensor = bridge::raf_backend::TryGetLtcTensor(self);
  if (!self_tensor) {
    bool sync_update = lazy_tensors::sys_util::GetEnvBool("RAF_TENSOR_UPDATE_SYNC", true);
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor = CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    LazyTensor::copy_(*dst_tensor, *self_tensor);
    bridge::ReplaceLtcTensor(dst, *dst_tensor);
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(const at::Tensor& self,
                                                      const at::Tensor& dst) {
  LTC_FN_COUNTER("raf::");
  auto dst_tensor = bridge::raf_backend::TryGetLtcTensor(dst);
  auto self_tensor = bridge::raf_backend::TryGetLtcTensor(self);
  if (!self_tensor) {
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor = CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    LTCTensorImpl* dest_impl = dynamic_cast<LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor().UpdateFromTensorOut(*self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor& LazyNativeFunctions::_index_put_impl_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices, const at::Tensor& values,
    bool accumulate, bool /* unsafe */) {
  LTC_FN_COUNTER("raf::");
  return index_put_(self, indices, values, accumulate);
}

at::Tensor LazyNativeFunctions::_log_softmax(const at::Tensor& self, int64_t dim,
                                             bool /* half_to_float */) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::log_softmax(bridge::raf_backend::GetLtcTensor(self), dim, c10::nullopt));
}

ir::NodePtr LogSoftmaxBackwardUseInOp(const ir::Value& grad_output, const ir::Value& output,
                                      int64_t dim) {
  return ir::MakeNode<ir::ops::LogSoftmaxBackwardUseIn>(
      grad_output, output, Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()));
}

LazyTensor log_softmax_backward(const LazyTensor& grad_output, const LazyTensor& output,
                                int64_t dim) {
  return bridge::raf_backend::CreateFrom(
      grad_output, LogSoftmaxBackwardUseInOp(grad_output.GetIrValue(), output.GetIrValue(), dim));
}

at::Tensor LazyNativeFunctions::_log_softmax_backward_data(const at::Tensor& grad_output,
                                                           const at::Tensor& output, int64_t dim,
                                                           at::ScalarType input_dtype) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      log_softmax_backward(bridge::raf_backend::GetLtcTensor(grad_output),
                           bridge::raf_backend::GetLtcTensor(output), dim));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::_pack_padded_sequence(
    const at::Tensor& input, const at::Tensor& lengths, bool batch_first) {
  LTC_FN_COUNTER("aten::");
  std::vector<at::Tensor> ratex_tensors = {lengths};
  auto cpu_tensors = bridge::LtcCreateTensorList(ratex_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor LazyNativeFunctions::where(const at::Tensor& condition, const at::Tensor& self,
                                      const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::where(bridge::raf_backend::GetLtcTensor(condition),
                                                     bridge::raf_backend::GetLtcTensor(self),
                                                     bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::_softmax(const at::Tensor& self, int64_t dim,
                                         bool /* half_to_float */) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK_EQ(dim, self.dim() - 1);
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax(bridge::raf_backend::GetLtcTensor(self), dim, c10::nullopt));
}

at::Tensor LazyNativeFunctions::_softmax_backward_data(const at::Tensor& grad_output,
                                                       const at::Tensor& output, int64_t dim,
                                                       at::ScalarType input_dtype) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax_backward(bridge::raf_backend::GetLtcTensor(grad_output),
                                   bridge::raf_backend::GetLtcTensor(output), dim));
}

at::Tensor LazyNativeFunctions::_trilinear(const at::Tensor& i1, const at::Tensor& i2,
                                           const at::Tensor& i3, at::IntArrayRef expand1,
                                           at::IntArrayRef expand2, at::IntArrayRef expand3,
                                           at::IntArrayRef sumdim, int64_t unroll_dim) {
  return FALLBACK_ATEN_OP(_trilinear, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

at::Tensor LazyNativeFunctions::_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("raf::");
  return view(self, size);
}

at::Tensor LazyNativeFunctions::abs(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::abs(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::acos(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::acos(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::acosh(const at::Tensor& self) {
  return FALLBACK_ATEN_OP(acosh, self);
}

at::Tensor LazyNativeFunctions::add(const at::Tensor& self, const at::Tensor& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor LazyNativeFunctions::add(const at::Tensor& self, const at::Scalar& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::add(xself, other, alpha, dtype);
                    });
}

at::Tensor LazyNativeFunctions::addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                                        const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcdiv(
      bridge::raf_backend::GetLtcTensor(self), value, bridge::raf_backend::GetLtcTensor(tensor1),
      bridge::raf_backend::GetLtcTensor(tensor2)));
}

at::Tensor LazyNativeFunctions::addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                                        const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcmul(
      bridge::raf_backend::GetLtcTensor(self), value, bridge::raf_backend::GetLtcTensor(tensor1),
      bridge::raf_backend::GetLtcTensor(tensor2)));
}

at::Tensor LazyNativeFunctions::addmm(const at::Tensor& self, const at::Tensor& mat1,
                                      const at::Tensor& mat2, const at::Scalar& beta,
                                      const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (beta.to<double>() != 1 || alpha.to<double>() != 1 || !at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat1) || !at::native::is_floating_point(mat2)) {
    return FALLBACK_ATEN_OP(addmm, self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::addmm(bridge::raf_backend::GetLtcTensor(mat1),
                        /*weight=*/bridge::raf_backend::GetLtcTensor(mat2),
                        /*bias=*/bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return self;
}

at::Tensor LazyNativeFunctions::all(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::all(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::all(bridge::raf_backend::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor LazyNativeFunctions::any(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::any(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::any(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::any(bridge::raf_backend::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor& LazyNativeFunctions::arange_out(const at::Scalar& start, const at::Scalar& end,
                                            const at::Scalar& step, at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor LazyNativeFunctions::argmax(const at::Tensor& self, c10::optional<int64_t> dim,
                                       bool keepdim) {
  LTC_FN_COUNTER("raf::");
  return dim ? bridge::AtenFromLtcTensor(
                   LazyTensor::argmax(bridge::raf_backend::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmax(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::argmin(const at::Tensor& self, c10::optional<int64_t> dim,
                                       bool keepdim) {
  LTC_FN_COUNTER("raf::");
  return dim ? bridge::AtenFromLtcTensor(
                   LazyTensor::argmin(bridge::raf_backend::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmin(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                           at::IntArrayRef stride,
                                           c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(self_tensor.shape(), xsize, xstride,
                                             storage_offset.value_or(0))) {
    return FALLBACK_ATEN_OP(as_strided, self, size, stride, storage_offset);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::as_strided(
      self_tensor, std::move(xsize), std::move(xstride), Helpers::I64Optional(storage_offset)));
}

const at::Tensor& LazyNativeFunctions::as_strided_(const at::Tensor& self, at::IntArrayRef size,
                                                   at::IntArrayRef stride,
                                                   c10::optional<int64_t> storage_offset) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (ir::ops::AsStrided::StrideIsSupported(self_tensor.shape(), xsize, xstride,
                                            storage_offset.value_or(0))) {
    LTC_FN_COUNTER("raf::");
    LazyTensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                            Helpers::I64Optional(storage_offset));
    return self;
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::as_strided_", 1);
  auto ratextens = bridge::LtcCreateTensorList({self});
  at::as_strided_(ratextens[0], size, stride, storage_offset);
  bridge::LtcUpdateTensors({self}, ratextens, {0});
  return self;
}

at::Tensor LazyNativeFunctions::asin(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::asin(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::asinh(const at::Tensor& self) {
  return FALLBACK_ATEN_OP(asinh, self);
}

at::Tensor LazyNativeFunctions::atan(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::atan(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::atanh(const at::Tensor& self) {
  return FALLBACK_ATEN_OP(atanh, self);
}

at::Tensor LazyNativeFunctions::atan2(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  // ratex::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return FALLBACK_ATEN_OP(atan2, self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor LazyNativeFunctions::avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                           bool ceil_mode, bool count_include_pad,
                                           c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("raf::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return FALLBACK_ATEN_OP(avg_pool2d, self, kernel_size, stride, padding, ceil_mode,
                            count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::avg_pool_nd(bridge::raf_backend::GetLtcTensor(self), /*spatial_dim_count=*/2,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor LazyNativeFunctions::avg_pool2d_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& self,
                                                    at::IntArrayRef kernel_size,
                                                    at::IntArrayRef stride, at::IntArrayRef padding,
                                                    bool ceil_mode, bool count_include_pad,
                                                    c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("raf::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return FALLBACK_ATEN_OP(avg_pool2d_backward, grad_output, self, kernel_size, stride, padding,
                            ceil_mode, count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor LazyNativeFunctions::avg_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                           bool ceil_mode, bool count_include_pad,
                                           c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("raf::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return FALLBACK_ATEN_OP(avg_pool3d, self, kernel_size, stride, padding, ceil_mode,
                            count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::avg_pool_nd(bridge::raf_backend::GetLtcTensor(self), /*spatial_dim_count=*/3,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor LazyNativeFunctions::avg_pool3d_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& self,
                                                    at::IntArrayRef kernel_size,
                                                    at::IntArrayRef stride, at::IntArrayRef padding,
                                                    bool ceil_mode, bool count_include_pad,
                                                    c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("raf::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return FALLBACK_ATEN_OP(avg_pool3d_backward, grad_output, self, kernel_size, stride, padding,
                            ceil_mode, count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor LazyNativeFunctions::baddbmm(const at::Tensor& self, const at::Tensor& batch1,
                                        const at::Tensor& batch2, const at::Scalar& beta,
                                        const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) || !at::native::is_floating_point(batch2)) {
    return FALLBACK_ATEN_OP(baddbmm, self, batch1, batch2, beta, alpha);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::baddbmm(
      bridge::raf_backend::GetLtcTensor(self), bridge::raf_backend::GetLtcTensor(batch1),
      bridge::raf_backend::GetLtcTensor(batch2), beta, alpha));
}

at::Tensor LazyNativeFunctions::bernoulli(const at::Tensor& self,
                                          c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return FALLBACK_ATEN_OP(bernoulli, self, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::bernoulli(self_tensor));
}

at::Tensor& LazyNativeFunctions::bernoulli_(at::Tensor& self, double p,
                                            c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::bernoulli_(self, p, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& LazyNativeFunctions::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                            c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::bernoulli_(self, p, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, bridge::raf_backend::GetLtcTensor(p));
  return self;
}

at::Tensor LazyNativeFunctions::binary_cross_entropy(const at::Tensor& self,
                                                     const at::Tensor& target,
                                                     const c10::optional<at::Tensor>& weight,
                                                     int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy(
      self_tensor, bridge::raf_backend::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor LazyNativeFunctions::binary_cross_entropy_backward(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), self_tensor,
      bridge::raf_backend::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor LazyNativeFunctions::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight, int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor& LazyNativeFunctions::bitwise_and_out(const at::Tensor& self, const at::Scalar& other,
                                                 at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_and_out(out_tensor, bridge::raf_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_and_out(const at::Tensor& self, const at::Tensor& other,
                                                 at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_and_out(out_tensor, bridge::raf_backend::GetLtcTensor(self),
                              bridge::raf_backend::GetLtcTensor(other));
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_not_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::bitwise_not_out(out_tensor, self_tensor);
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_or_out(const at::Tensor& self, const at::Scalar& other,
                                                at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::raf_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_or_out(const at::Tensor& self, const at::Tensor& other,
                                                at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::raf_backend::GetLtcTensor(self),
                             bridge::raf_backend::GetLtcTensor(other));
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_xor_out(const at::Tensor& self, const at::Scalar& other,
                                                 at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::raf_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& LazyNativeFunctions::bitwise_xor_out(const at::Tensor& self, const at::Tensor& other,
                                                 at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::raf_backend::GetLtcTensor(self),
                              bridge::raf_backend::GetLtcTensor(other));
  return out;
}

at::Tensor LazyNativeFunctions::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(mat2)) {
    return FALLBACK_ATEN_OP(bmm, self, mat2);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::bmm(bridge::raf_backend::GetLtcTensor(self),
                                                   bridge::raf_backend::GetLtcTensor(mat2)));
}

at::Tensor LazyNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cat(bridge::raf_backend::GetLtcTensors(tensors), dim));
}

at::Tensor LazyNativeFunctions::ceil(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ceil(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::cholesky(const at::Tensor& self, bool upper) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cholesky(bridge::raf_backend::GetLtcTensor(self), upper));
}

at::Tensor LazyNativeFunctions::clamp(const at::Tensor& self, const c10::optional<at::Scalar>& min,
                                      const c10::optional<at::Scalar>& max) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::raf_backend::GetLtcTensor(self), min, max));
}

at::Tensor LazyNativeFunctions::clamp(const at::Tensor& self, const c10::optional<at::Tensor>& min,
                                      const c10::optional<at::Tensor>& max) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::raf_backend::GetLtcTensor(self), min, max));
}

at::Tensor LazyNativeFunctions::clamp_max(const at::Tensor& self, const at::Scalar& max) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::raf_backend::GetLtcTensor(self), c10::nullopt, max));
}

at::Tensor& LazyNativeFunctions::clamp_max_out(const at::Tensor& self, const at::Tensor& max,
                                               at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::raf_backend::GetLtcTensor(self), c10::nullopt, max);
  return out;
}

at::Tensor LazyNativeFunctions::clamp_min(const at::Tensor& self, const at::Scalar& min) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::raf_backend::GetLtcTensor(self), min, c10::nullopt));
}

at::Tensor& LazyNativeFunctions::clamp_min_out(const at::Tensor& self, const at::Tensor& min,
                                               at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::raf_backend::GetLtcTensor(self), min, c10::nullopt);
  return out;
}

at::Tensor LazyNativeFunctions::clone(const at::Tensor& self,
                                      c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor));
}

at::Tensor LazyNativeFunctions::constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad,
                                                const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::constant_pad_nd(
      bridge::raf_backend::GetLtcTensor(self), Helpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor LazyNativeFunctions::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  LTC_FN_COUNTER("raf::");
  if (IsDefined(bias)) {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::raf_backend::GetLtcTensor(input), bridge::raf_backend::GetLtcTensor(weight),
        bridge::raf_backend::GetLtcTensor(*bias), Helpers::I64List(stride),
        Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  } else {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::raf_backend::GetLtcTensor(input), bridge::raf_backend::GetLtcTensor(weight),
        Helpers::I64List(stride), Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  }
}

// This functions covers the whole convolution backward lowering.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("raf::");
  auto gradients = LazyTensor::convolution_backward_overrideable(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(input),
      bridge::raf_backend::GetLtcTensor(weight), Helpers::I64List(stride),
      Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
      Helpers::I64List(output_padding), groups);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients)) : at::Tensor(),
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients)) : at::Tensor(),
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients)) : at::Tensor());
}

at::Tensor LazyNativeFunctions::cos(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::cos(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::cosh(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::cosh(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::cross(const at::Tensor& self, const at::Tensor& other,
                                      c10::optional<int64_t> dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::cross(bridge::raf_backend::GetLtcTensor(self),
                                                     bridge::raf_backend::GetLtcTensor(other),
                                                     Helpers::I64Optional(dim)));
}

at::Tensor LazyNativeFunctions::cumprod(const at::Tensor& self, int64_t dim,
                                        c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  c10::optional<at::ScalarType> promoted_dtype = PromoteIntegralType(self_tensor.dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // Reduce-window does not support S64 mode.
    return FALLBACK_ATEN_OP(cumprod, self, dim, dtype);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor LazyNativeFunctions::cumsum(const at::Tensor& self, int64_t dim,
                                       c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  if (IsOperationOnType(dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // Reduce-window does not support S64 mode.
    return FALLBACK_ATEN_OP(cumsum, self, dim, dtype);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::cumsum(self_tensor, dim, dtype));
}

at::Tensor LazyNativeFunctions::diag(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diag(bridge::raf_backend::GetLtcTensor(self), diagonal));
}

at::Tensor LazyNativeFunctions::diagonal(const at::Tensor& self, int64_t offset, int64_t dim1,
                                         int64_t dim2) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diagonal(bridge::raf_backend::GetLtcTensor(self), offset, dim1, dim2));
}

at::Tensor LazyNativeFunctions::div(const at::Tensor& self, const at::Tensor& other) {
  return div(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor LazyNativeFunctions::div(const at::Tensor& self, const at::Tensor& other,
                                    c10::optional<c10::string_view> rounding_mode) {
  LTC_FN_COUNTER("raf::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, UnwrapNumber(other, dtype));
  return bridge::AtenFromLtcTensor(
      LazyTensor::div(operands.first, operands.second, rounding_mode, dtype));
}

at::Tensor LazyNativeFunctions::div(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::div(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::dot(const at::Tensor& self, const at::Tensor& tensor) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK_EQ(self.dim(), 1) << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  LTC_CHECK_EQ(tensor.dim(), 1) << "dot: Expected 1-D argument tensor, but got " << tensor.dim()
                                << "-D";
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(tensor)) {
    return FALLBACK_ATEN_OP(dot, self, tensor);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::matmul(bridge::raf_backend::GetLtcTensor(self),
                                                      bridge::raf_backend::GetLtcTensor(tensor)));
}

at::Tensor LazyNativeFunctions::dropout(const at::Tensor& input, double p, bool train) {
  return aten_autograd_ops::Dropout::apply(input, p, train);
}

at::Tensor LazyNativeFunctions::elu(const at::Tensor& self, const at::Scalar& alpha,
                                    const at::Scalar& scale, const at::Scalar& input_scale) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu(bridge::raf_backend::GetLtcTensor(self), alpha, scale, input_scale));
}

at::Tensor LazyNativeFunctions::elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha,
                                             const at::Scalar& scale, const at::Scalar& input_scale,
                                             bool self, const at::Tensor& self_or_result) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu_backward(bridge::raf_backend::GetLtcTensor(grad_output), alpha, scale,
                               input_scale, bridge::raf_backend::GetLtcTensor(self_or_result)));
}

at::Tensor LazyNativeFunctions::embedding(const at::Tensor& weight, const at::Tensor& indices,
                                          int64_t padding_idx, bool scale_grad_by_freq,
                                          bool sparse) {
  LTC_FN_COUNTER("raf::");
  if (scale_grad_by_freq || sparse || padding_idx != -1) {
    RATEX_VLOG(3) << "Unsupported parameters - Falling back to CPU (currently sparse, "
                     "scale_grad_by_freq, and padding are not support)";
    return FALLBACK_ATEN_OP(embedding, weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
  LazyTensor weight_tensor;
  LazyTensor indices_tensor;
  auto weight_xtensor = bridge::raf_backend::TryGetLtcTensor(weight);
  if (!weight_xtensor) {
    indices_tensor = bridge::raf_backend::GetLtcTensor(indices);
    weight_tensor = bridge::GetOrCreateLtcTensor(weight, indices_tensor.GetDevice());
  } else {
    weight_tensor = *weight_xtensor;
    indices_tensor = bridge::GetOrCreateLtcTensor(indices, weight_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(LazyTensor::embedding(weight_tensor, indices_tensor, padding_idx,
                                                         scale_grad_by_freq, sparse));
}

at::Tensor LazyNativeFunctions::embedding_dense_backward(const at::Tensor& grad_output,
                                                         const at::Tensor& indices,
                                                         int64_t num_weights, int64_t padding_idx,
                                                         bool scale_grad_by_freq) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::embedding_dense_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor LazyNativeFunctions::empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory,
                                      c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("raf::");
  // PT empty*() are optimizations to avoid initializing the data when it is
  // known it will be completely rewritten. But since for us doing a zero*()
  // does not actually end up doing any memory initialization, we use that and
  // avoid going to CPU for it. A common PT pattern is indeed doing empty()
  // plus s_copy_().
  return bridge::AtenFromLtcTensor(LazyTensor::full(
      Helpers::I64List(size), 0, GetLtcDeviceOrCurrent(device), GetScalarTypeOrFloat(dtype)));
}

at::Tensor LazyNativeFunctions::empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                              c10::optional<at::ScalarType> dtype,
                                              c10::optional<at::Layout> layout,
                                              c10::optional<at::Device> device,
                                              c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("raf::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor LazyNativeFunctions::eq(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::eq(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::eq(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::eq(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::erf(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::erf(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::erfc(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::erfc(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::erfinv(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::erfinv(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::exp(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::exp(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::expand(const at::Tensor& self, at::IntArrayRef size,
                                       bool implicit) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::expand(bridge::raf_backend::GetLtcTensor(self),
                                                      lazy_tensors::util::ToVector<int64_t>(size)));
}

at::Tensor LazyNativeFunctions::expm1(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::expm1(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor& LazyNativeFunctions::exponential_(at::Tensor& self, double lambd,
                                              c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return FALLBACK_ATEN_OP(exponential_, self, lambd, generator);
  }
  LTC_CHECK_GE(lambd, 0.0);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& LazyNativeFunctions::eye_out(int64_t n, at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& LazyNativeFunctions::eye_out(int64_t n, int64_t m, at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& LazyNativeFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("raf::");
    LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
    LazyTensor::fill_(self_tensor, value);
    return self;
  }
  return AtenRAFTypeDefault::fill_(self, value);
}

at::Tensor& LazyNativeFunctions::fill_(at::Tensor& self, const at::Tensor& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("raf::");
    LTC_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                                 << "value tensor, but got tensor "
                                 << "with " << value.dim() << " dimension(s).";
    return fill_(self, value.item());
  }
  return AtenRAFTypeDefault::fill_(self, value);
}

at::Tensor LazyNativeFunctions::flip(const at::Tensor& self, at::IntArrayRef dims) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::flip(bridge::raf_backend::GetLtcTensor(self), Helpers::I64List(dims)));
}

at::Tensor LazyNativeFunctions::floor(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::floor(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::fmod(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor LazyNativeFunctions::fmod(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, other, dtype);
                    });
}

at::Tensor LazyNativeFunctions::frac(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::frac(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::gather(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                                       bool /* sparse_grad */) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::gather(bridge::raf_backend::GetLtcTensor(self), dim,
                                                      bridge::raf_backend::GetLtcTensor(index)));
}

at::Tensor LazyNativeFunctions::ge(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ge(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::ge(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ge(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::gelu(const at::Tensor& self, c10::string_view approximate) {
  LTC_FN_COUNTER("raf::");
  auto gelu_type = at::native::get_gelutype_enum(approximate);
  LTC_CHECK_EQ(gelu_type, at::native::GeluType::None) << "Not supported GeLU approximation yet";
  return bridge::AtenFromLtcTensor(LazyTensor::gelu(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::gelu_backward(const at::Tensor& grad, const at::Tensor& self,
                                              c10::string_view approximate) {
  LTC_FN_COUNTER("raf::");
  auto gelu_type = at::native::get_gelutype_enum(approximate);
  LTC_CHECK_EQ(gelu_type, at::native::GeluType::None) << "Not supported GeLU approximation yet";
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::raf_backend::GetLtcTensor(grad), bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::ger(const at::Tensor& self, const at::Tensor& vec2) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ger(bridge::raf_backend::GetLtcTensor(self),
                                                   bridge::raf_backend::GetLtcTensor(vec2)));
}

at::Tensor LazyNativeFunctions::gt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::gt(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::gt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::gt(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::hardshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardshrink(bridge::raf_backend::GetLtcTensor(self), lambda));
}

at::Tensor LazyNativeFunctions::hardsigmoid(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardsigmoid(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::hardsigmoid_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardsigmoid_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::hardshrink_backward(const at::Tensor& grad_out,
                                                    const at::Tensor& self,
                                                    const at::Scalar& lambda) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardshrink_backward(bridge::raf_backend::GetLtcTensor(grad_out),
                                      bridge::raf_backend::GetLtcTensor(self), lambda));
}

at::Tensor LazyNativeFunctions::hardtanh(const at::Tensor& self, const at::Scalar& min_val,
                                         const at::Scalar& max_val) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::raf_backend::GetLtcTensor(self), min_val, max_val));
}

at::Tensor LazyNativeFunctions::hardtanh_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, const at::Scalar& min_val,
                                                  const at::Scalar& max_val) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardtanh_backward(bridge::raf_backend::GetLtcTensor(grad_output),
                                    bridge::raf_backend::GetLtcTensor(self), min_val, max_val));
}

at::Tensor LazyNativeFunctions::index(const at::Tensor& self,
                                      const c10::List<c10::optional<at::Tensor>>& indices) {
  LTC_FN_COUNTER("raf::");
  CanonicalIndexInfo canonical_index_info = GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromLtcTensor(
      LazyTensor::index(bridge::raf_backend::GetLtcTensor(canonical_index_info.base),
                        bridge::raf_backend::GetLtcTensors(canonical_index_info.indices),
                        canonical_index_info.start_dim));
}

at::Tensor& LazyNativeFunctions::index_add_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                            const at::Tensor& source, const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK_EQ(alpha.toFloat(), 1.0) << "Not support alpha != 1.0";
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::index_add_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index),
                         bridge::raf_backend::GetLtcTensor(source));
  return self;
}

at::Tensor& LazyNativeFunctions::index_copy_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                             const at::Tensor& source) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::index_copy_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index),
                          bridge::raf_backend::GetLtcTensor(source));
  return self;
}

at::Tensor& LazyNativeFunctions::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                             const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index), value);
  return self;
}

at::Tensor& LazyNativeFunctions::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                             const at::Tensor& value) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index),
                          bridge::raf_backend::GetLtcTensor(value));
  return self;
}

at::Tensor& LazyNativeFunctions::index_put_(at::Tensor& self,
                                            const c10::List<c10::optional<at::Tensor>>& indices,
                                            const at::Tensor& values, bool accumulate) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK(self.scalar_type() == values.scalar_type());
  CanonicalIndexInfo canonical_index_info = GetCanonicalIndexInfo(self, indices);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::index_put_(self_tensor, bridge::raf_backend::GetLtcTensor(canonical_index_info.base),
                         bridge::raf_backend::GetLtcTensors(canonical_index_info.indices),
                         canonical_index_info.start_dim, bridge::raf_backend::GetLtcTensor(values),
                         accumulate, canonical_index_info.result_permutation);
  return self;
}

at::Tensor LazyNativeFunctions::index_select(const at::Tensor& self, int64_t dim,
                                             const at::Tensor& index) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::index_select(
      bridge::raf_backend::GetLtcTensor(self), dim, bridge::raf_backend::GetLtcTensor(index)));
}

at::Tensor LazyNativeFunctions::inverse(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::inverse(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::isnan(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::isnan(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::kl_div(const at::Tensor& self, const at::Tensor& target,
                                       int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("raf::");
  return at::native::kl_div(self, target, reduction, log_target);
}

at::Tensor LazyNativeFunctions::kl_div_backward(const at::Tensor& grad_output,
                                                const at::Tensor& self, const at::Tensor& target,
                                                int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::kl_div_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      bridge::raf_backend::GetLtcTensor(target), reduction, log_target));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::kthvalue(const at::Tensor& self, int64_t k,
                                                                 int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::kthvalue(bridge::raf_backend::GetLtcTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor LazyNativeFunctions::l1_loss(const at::Tensor& self, const at::Tensor& target,
                                        int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss(bridge::raf_backend::GetLtcTensor(self),
                                                       bridge::raf_backend::GetLtcTensor(target),
                                                       reduction));
}

at::Tensor LazyNativeFunctions::l1_loss_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self, const at::Tensor& target,
                                                 int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      bridge::raf_backend::GetLtcTensor(target), reduction));
}

at::Tensor LazyNativeFunctions::le(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::le(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::le(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::le(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::leaky_relu(const at::Tensor& self,
                                           const at::Scalar& negative_slope) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::leaky_relu(bridge::raf_backend::GetLtcTensor(self), negative_slope.to<double>()));
}

at::Tensor LazyNativeFunctions::leaky_relu_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& self,
                                                    const at::Scalar& negative_slope,
                                                    bool self_is_result) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK(!self_is_result || negative_slope.to<double>() > 0.0);
  return bridge::AtenFromLtcTensor(LazyTensor::leaky_relu_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      negative_slope.to<double>()));
}

at::Tensor LazyNativeFunctions::log(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::log(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::log10(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(bridge::raf_backend::GetLtcTensor(self),
                                                        ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor LazyNativeFunctions::log1p(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::log1p(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::log2(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(bridge::raf_backend::GetLtcTensor(self),
                                                        ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor LazyNativeFunctions::log_sigmoid_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& self,
                                                     const at::Tensor& buffer) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_sigmoid_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      bridge::raf_backend::GetLtcTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::log_sigmoid_forward(
    const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  auto result_tuple = LazyTensor::log_sigmoid_forward(bridge::raf_backend::GetLtcTensor(self));
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromLtcTensor(std::get<1>(result_tuple)));
}

at::Tensor LazyNativeFunctions::logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                                          bool keepdim) {
  return FALLBACK_ATEN_OP(logsumexp, self, dim, keepdim);
}

at::Tensor LazyNativeFunctions::logdet(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::logdet(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::logical_or(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::logical_or(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::lt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::lt(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::lt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::lt(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor& LazyNativeFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                              const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::masked_fill_(self_tensor, bridge::raf_backend::GetLtcTensor(mask), value);
  return self;
}

at::Tensor& LazyNativeFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                              const at::Tensor& value) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor& LazyNativeFunctions::masked_scatter_(at::Tensor& self, const at::Tensor& mask,
                                                 const at::Tensor& source) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::masked_scatter_(self_tensor, bridge::raf_backend::GetLtcTensor(mask),
                              bridge::raf_backend::GetLtcTensor(source));
  return self;
}

at::Tensor LazyNativeFunctions::masked_select(const at::Tensor& self, const at::Tensor& mask) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return FALLBACK_ATEN_OP(masked_select, self, mask);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::masked_select(self_tensor, bridge::raf_backend::GetLtcTensor(mask)));
}

at::Tensor LazyNativeFunctions::max(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::max(bridge::raf_backend::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max(const at::Tensor& self, int64_t dim,
                                                            bool keepdim) {
  LTC_FN_COUNTER("raf::");
  auto outputs = LazyTensor::max(bridge::raf_backend::GetLtcTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor LazyNativeFunctions::maximum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::max(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> LazyNativeFunctions::max_out(const at::Tensor& self,
                                                                  int64_t dim, bool keepdim,
                                                                  at::Tensor& max,
                                                                  at::Tensor& max_values) {
  LTC_FN_COUNTER("raf::");
  LazyTensor max_tensor = bridge::raf_backend::GetLtcTensor(max);
  LazyTensor max_values_tensor = bridge::raf_backend::GetLtcTensor(max_values);
  LazyTensor::max_out(max_tensor, max_values_tensor, bridge::raf_backend::GetLtcTensor(self), dim,
                      keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor LazyNativeFunctions::max_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                           at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("raf::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(self, kernel_size, stride, padding,
                                                             dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("raf::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return FALLBACK_ATEN_OP(max_pool2d_with_indices, self, kernel_size, stride, padding, dilation,
                            ceil_mode);
  }
  auto outputs =
      LazyTensor::max_pool_nd(bridge::raf_backend::GetLtcTensor(self), /*spatial_dim_count=*/2,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor LazyNativeFunctions::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("raf::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return FALLBACK_ATEN_OP(max_pool2d_with_indices_backward, grad_output, self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));
}

at::Tensor LazyNativeFunctions::max_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                           at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("raf::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(self, kernel_size, stride, padding,
                                                             dilation, ceil_mode);
}

at::Tensor LazyNativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("raf::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return FALLBACK_ATEN_OP(max_pool3d_with_indices_backward, grad_output, self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("raf::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return FALLBACK_ATEN_OP(max_pool3d_with_indices, self, kernel_size, stride, padding, dilation,
                            ceil_mode);
  }
  auto outputs =
      LazyTensor::max_pool_nd(bridge::raf_backend::GetLtcTensor(self), /*spatial_dim_count=*/3,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor LazyNativeFunctions::max_unpool2d(const at::Tensor& self, const at::Tensor& indices,
                                             at::IntArrayRef output_size) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::raf_backend::GetLtcTensor(self), bridge::raf_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<int64_t>(output_size)));
}


at::Tensor LazyNativeFunctions::max_unpool3d(const at::Tensor& self, const at::Tensor& indices,
                                             at::IntArrayRef output_size, at::IntArrayRef stride,
                                             at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::raf_backend::GetLtcTensor(self), bridge::raf_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<int64_t>(output_size)));
}


at::Tensor LazyNativeFunctions::mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::mean(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor LazyNativeFunctions::mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                     c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::mean(bridge::raf_backend::GetLtcTensor(self),
                                                    lazy_tensors::util::ToVector<int64_t>(dim),
                                                    /*keep_reduced_dimensions=*/keepdim, dtype));
}

at::Tensor LazyNativeFunctions::min(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::min(bridge::raf_backend::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::min(const at::Tensor& self, int64_t dim,
                                                            bool keepdim) {
  return AtenRAFTypeDefault::min(self, dim, keepdim);
}

at::Tensor LazyNativeFunctions::minimum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::min(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> LazyNativeFunctions::min_out(const at::Tensor& self,
                                                                  int64_t dim, bool keepdim,
                                                                  at::Tensor& min,
                                                                  at::Tensor& min_indices) {
  LTC_FN_COUNTER("raf::");
  LazyTensor min_tensor = bridge::raf_backend::GetLtcTensor(min);
  LazyTensor min_indices_tensor = bridge::raf_backend::GetLtcTensor(min_indices);
  LazyTensor::min_out(min_tensor, min_indices_tensor, bridge::raf_backend::GetLtcTensor(self), dim,
                      keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor LazyNativeFunctions::mm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(mat2)) {
    return FALLBACK_ATEN_OP(mm, self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::mm(/*input=*/bridge::raf_backend::GetLtcTensor(self),
                     /*weight=*/bridge::raf_backend::GetLtcTensor(mat2)));
}

at::Tensor LazyNativeFunctions::mse_loss(const at::Tensor& self, const at::Tensor& target,
                                         int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss(bridge::raf_backend::GetLtcTensor(self),
                                                        bridge::raf_backend::GetLtcTensor(target),
                                                        reduction));
}

at::Tensor LazyNativeFunctions::mse_loss_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, const at::Tensor& target,
                                                  int64_t reduction) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      bridge::raf_backend::GetLtcTensor(target), reduction));
}

at::Tensor LazyNativeFunctions::mul(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::mul(xself, xother, dtype);
                    });
}

at::Tensor LazyNativeFunctions::mul(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::mul(xself, other, dtype);
                    });
}

at::Tensor LazyNativeFunctions::mv(const at::Tensor& self, const at::Tensor& vec) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(vec)) {
    return FALLBACK_ATEN_OP(mv, self, vec);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::mv(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(vec)));
}

at::Tensor& LazyNativeFunctions::mv_out(const at::Tensor& self, const at::Tensor& vec,
                                        at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  // ratex::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(vec)) {
    return FALLBACK_ATEN_OP(mv_out, self, vec, out);
  }
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor::mv_out(out_tensor, bridge::raf_backend::GetLtcTensor(self),
                     bridge::raf_backend::GetLtcTensor(vec));
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LazyNativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
  LTC_FN_COUNTER("raf::");
  LazyTensor input_tensor = bridge::raf_backend::GetLtcTensor(input);
  const Device& device = input_tensor.GetDevice();
  LazyTensor running_mean_tensor = bridge::GetOrCreateLtcTensor(running_mean, device);
  LazyTensor running_var_tensor = bridge::GetOrCreateLtcTensor(running_var, device);
  auto outputs = LazyTensor::native_batch_norm(
      bridge::raf_backend::GetLtcTensor(input), bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(bias, device), running_mean_tensor, running_var_tensor, training,
      momentum, eps);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LazyNativeFunctions::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean, const c10::optional<at::Tensor>& save_invstd,
    bool train, double eps, std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("raf::");
  LazyTensor grad_out_tensor = bridge::raf_backend::GetLtcTensor(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = LazyTensor::native_batch_norm_backward(
      bridge::raf_backend::GetLtcTensor(grad_out), bridge::raf_backend::GetLtcTensor(input),
      bridge::GetOrCreateLtcTensor(weight, device), bridge::GetOrCreateLtcTensor(save_mean, device),
      bridge::GetOrCreateLtcTensor(save_invstd, device), train, eps);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients)) : undefined,
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients)) : undefined,
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients)) : undefined);
}

at::Tensor LazyNativeFunctions::ne(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ne(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::ne(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::ne(bridge::raf_backend::GetLtcTensor(self),
                                                  bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::neg(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  LTC_CHECK(self.scalar_type() != at::kBool)
      << "Negation, the `-` operator, on a bool tensor is not supported. If "
         "you are trying to invert a mask, use the `~` or `logical_not()` "
         "operator instead.";
  return bridge::AtenFromLtcTensor(LazyTensor::neg(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::nll_loss2d_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& self,
                                                    const at::Tensor& target,
                                                    const c10::optional<at::Tensor>& weight,
                                                    int64_t reduction, int64_t ignore_index,
                                                    const at::Tensor& total_weight) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor = bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::nll_loss2d_backward(bridge::raf_backend::GetLtcTensor(grad_output), self_tensor,
                                      bridge::raf_backend::GetLtcTensor(target), weight_tensor,
                                      reduction, ignore_index, total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor total_weight = LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss2d(
          self_tensor, bridge::raf_backend::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()), reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor LazyNativeFunctions::nll_loss_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, const at::Tensor& target,
                                                  const c10::optional<at::Tensor>& weight,
                                                  int64_t reduction, int64_t ignore_index,
                                                  const at::Tensor& total_weight) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor = bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::nll_loss_backward(bridge::raf_backend::GetLtcTensor(grad_output), self_tensor,
                                    bridge::raf_backend::GetLtcTensor(target), weight_tensor,
                                    reduction, ignore_index, total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor total_weight = LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss(
          self_tensor, bridge::raf_backend::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()), reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor LazyNativeFunctions::nonzero(const at::Tensor& self) {
  return FALLBACK_ATEN_OP(nonzero, self);
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::ScalarType dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::norm(
      self_tensor, p, dtype, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const at::Scalar& p) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::norm(self_tensor, p, self_tensor.dtype(),
                       lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
                       /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::norm(self_tensor, p, dtype, dim, keepdim));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::norm(self_tensor, p, self_tensor.dtype(), dim, keepdim));
}

at::Tensor LazyNativeFunctions::normal(const at::Tensor& mean, double std,
                                       c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(bridge::raf_backend::GetLtcTensor(mean), std));
}

at::Tensor LazyNativeFunctions::normal(double mean, const at::Tensor& std,
                                       c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(mean, bridge::raf_backend::GetLtcTensor(std)));
}

at::Tensor LazyNativeFunctions::normal(const at::Tensor& mean, const at::Tensor& std,
                                       c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::normal(bridge::raf_backend::GetLtcTensor(mean),
                                                      bridge::raf_backend::GetLtcTensor(std)));
}

at::Tensor& LazyNativeFunctions::normal_(at::Tensor& self, double mean, double std,
                                         c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return FALLBACK_ATEN_OP(normal_, self, mean, std, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor LazyNativeFunctions::permute(const at::Tensor& self, at::IntArrayRef dims) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
}

at::Tensor LazyNativeFunctions::pow(const at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_COUNTER("raf::");
  // ratex::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenRAFTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(bridge::raf_backend::GetLtcTensor(self), exponent));
}

at::Tensor LazyNativeFunctions::pow(const at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("raf::");
  // ratex::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenRAFTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::pow(bridge::raf_backend::GetLtcTensor(self),
                                                   bridge::raf_backend::GetLtcTensor(exponent)));
}

at::Tensor LazyNativeFunctions::pow(const at::Scalar& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("raf::");
  // ratex::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return AtenRAFTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(self, bridge::raf_backend::GetLtcTensor(exponent)));
}

at::Tensor LazyNativeFunctions::prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::prod(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor LazyNativeFunctions::prod(const at::Tensor& self, int64_t dim, bool keepdim,
                                     c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::prod(bridge::raf_backend::GetLtcTensor(self), {dim}, keepdim,
                       PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor& LazyNativeFunctions::put_(at::Tensor& self, const at::Tensor& index,
                                      const at::Tensor& source, bool accumulate) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::put_(self_tensor, bridge::raf_backend::GetLtcTensor(index),
                   bridge::raf_backend::GetLtcTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::qr(const at::Tensor& self, bool some) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::qr(bridge::raf_backend::GetLtcTensor(self), some);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

// The value generated should be within (from, to].
at::Tensor& LazyNativeFunctions::random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                                         c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::random_(self, from, to, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  at::ScalarType dtype = self_tensor.dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  int64_t to_val = (to) ? *to : GetIntegerUpperLimitForType(dtype) + inc;
  LTC_CHECK_LE(from, to_val);
  CheckRangeValues(self_tensor.dtype(), from, to_val - 1);
  LazyTensor::random_(self_tensor, from, to_val);
  return self;
}

// The value generated should be in (0, to].
at::Tensor& LazyNativeFunctions::random_(at::Tensor& self, int64_t to,
                                         c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return AtenRAFTypeDefault::random_(self, to, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_CHECK_GT(to, 0);
  CheckRangeValues(self_tensor.dtype(), 0, to - 1);
  LazyTensor::random_(self_tensor, 0, to);
  return self;
}

// The value generated should be in (self_type_min, self_type_max).
at::Tensor& LazyNativeFunctions::random_(at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return FALLBACK_ATEN_OP(random_, self, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  at::ScalarType dtype = self_tensor.dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  LazyTensor::random_(self_tensor, 0, GetIntegerUpperLimitForType(dtype) + inc);
  return self;
}

at::Tensor LazyNativeFunctions::reciprocal(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::reciprocal(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::reflection_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::reflection_pad2d(
      bridge::raf_backend::GetLtcTensor(self), lazy_tensors::util::ToVector<int64_t>(padding)));
}

at::Tensor LazyNativeFunctions::reflection_pad2d_backward(const at::Tensor& grad_output,
                                                          const at::Tensor& self,
                                                          at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::reflection_pad2d_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      lazy_tensors::util::ToVector<int64_t>(padding)));
}

at::Tensor LazyNativeFunctions::relu(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::relu(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::remainder(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::remainder(bridge::raf_backend::GetLtcTensor(self),
                                                         bridge::raf_backend::GetLtcTensor(other)));
}

at::Tensor LazyNativeFunctions::remainder(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::remainder(bridge::raf_backend::GetLtcTensor(self), other));
}

at::Tensor LazyNativeFunctions::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  return FALLBACK_ATEN_OP(repeat, self, repeats);
}

at::Tensor LazyNativeFunctions::replication_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d(
      bridge::raf_backend::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor LazyNativeFunctions::replication_pad1d_backward(const at::Tensor& grad_output,
                                                           const at::Tensor& self,
                                                           at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

at::Tensor LazyNativeFunctions::replication_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d(
      bridge::raf_backend::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor LazyNativeFunctions::replication_pad2d_backward(const at::Tensor& grad_output,
                                                           const at::Tensor& self,
                                                           at::IntArrayRef padding) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

const at::Tensor& LazyNativeFunctions::resize_(const at::Tensor& self, at::IntArrayRef size,
                                               c10::optional<at::MemoryFormat> memory_format) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("raf::");
    LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
    LazyTensor::resize_(self_tensor, Helpers::I64List(size));
    return self;
  }
  return FALLBACK_ATEN_OP(resize_, self, size, memory_format);
}

at::Tensor LazyNativeFunctions::round(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::round(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::rrelu_with_noise(const at::Tensor& self, const at::Tensor& noise,
                                                 const at::Scalar& lower, const at::Scalar& upper,
                                                 bool training,
                                                 c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    LTC_CHECK_EQ(training, false);
    return FALLBACK_ATEN_OP(rrelu_with_noise, self, noise, lower, upper, training, generator);
  }
  LazyTensor noise_tensor = bridge::raf_backend::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise(
      bridge::raf_backend::GetLtcTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor LazyNativeFunctions::rrelu_with_noise_backward(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& noise,
    const at::Scalar& lower, const at::Scalar& upper, bool training, bool self_is_result) {
  LTC_FN_COUNTER("raf::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  LTC_CHECK(!self_is_result || negative_slope > 0.0);
  LazyTensor noise_tensor = bridge::raf_backend::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor LazyNativeFunctions::rsqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::rsqrt(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::rsub(const at::Tensor& self, const at::Tensor& other,
                                     const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::rsub(xself, xother, alpha, dtype);
                    });
}

at::Tensor LazyNativeFunctions::rsub(const at::Tensor& self, const at::Scalar& other,
                                     const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return bridge::AtenFromLtcTensor(
      LazyTensor::rsub(bridge::raf_backend::GetLtcTensor(self), other, alpha));
}

at::Tensor& LazyNativeFunctions::scatter_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                          const at::Tensor& src) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::scatter_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index),
                       bridge::raf_backend::GetLtcTensor(src));
  return self;
}

at::Tensor& LazyNativeFunctions::scatter_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                          const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::scatter_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index), value);
  return self;
}

at::Tensor& LazyNativeFunctions::scatter_add_(at::Tensor& self, int64_t dim,
                                              const at::Tensor& index, const at::Tensor& src) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::scatter_add_(self_tensor, dim, bridge::raf_backend::GetLtcTensor(index),
                           bridge::raf_backend::GetLtcTensor(src));
  return self;
}

at::Tensor LazyNativeFunctions::select(const at::Tensor& self, int64_t dim, int64_t index) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::select(bridge::raf_backend::GetLtcTensor(self), dim, index));
}

at::Tensor& LazyNativeFunctions::silu_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("raf::");
  LazyTensor out_tensor = bridge::raf_backend::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::silu_out(self_tensor, out_tensor);
  return out;
}

at::Tensor LazyNativeFunctions::sigmoid(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sigmoid(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::sigmoid_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& output) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sigmoid_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(output)));
}

at::Tensor LazyNativeFunctions::sign(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sign(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::sin(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sin(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::sinh(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sinh(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::slice(const at::Tensor& self, int64_t dim,
                                      c10::optional<int64_t> start, c10::optional<int64_t> end,
                                      int64_t step) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  return bridge::AtenFromLtcTensor(
      LazyTensor::slice(bridge::raf_backend::GetLtcTensor(self), dim, start_val, end_val, step));
}

at::Tensor LazyNativeFunctions::smooth_l1_loss(const at::Tensor& self, const at::Tensor& target,
                                               int64_t reduction, double beta) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::smooth_l1_loss(bridge::raf_backend::GetLtcTensor(self),
                                 bridge::raf_backend::GetLtcTensor(target), reduction, beta));
}

at::Tensor LazyNativeFunctions::smooth_l1_loss_backward(const at::Tensor& grad_output,
                                                        const at::Tensor& self,
                                                        const at::Tensor& target, int64_t reduction,
                                                        double beta) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::smooth_l1_loss_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      bridge::raf_backend::GetLtcTensor(target), reduction, beta));
}

at::Tensor LazyNativeFunctions::softplus(const at::Tensor& self, const at::Scalar& beta,
                                         const at::Scalar& threshold) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softplus(bridge::raf_backend::GetLtcTensor(self), beta, threshold));
}

at::Tensor LazyNativeFunctions::softplus_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, const at::Scalar& beta,
                                                  const at::Scalar& threshold) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softplus_backward(bridge::raf_backend::GetLtcTensor(grad_output),
                                    bridge::raf_backend::GetLtcTensor(self), beta, threshold));
}

at::Tensor LazyNativeFunctions::softshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink(bridge::raf_backend::GetLtcTensor(self), lambda));
}

at::Tensor LazyNativeFunctions::softshrink_backward(const at::Tensor& grad_out,
                                                    const at::Tensor& self,
                                                    const at::Scalar& lambda) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink_backward(bridge::raf_backend::GetLtcTensor(grad_out),
                                      bridge::raf_backend::GetLtcTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::sort(const at::Tensor& self, int64_t dim,
                                                             bool descending) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::topk(bridge::raf_backend::GetLtcTensor(self), self.size(dim), dim,
                                  descending, true);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

std::vector<at::Tensor> LazyNativeFunctions::split(const at::Tensor& self, int64_t split_size,
                                                   int64_t dim) {
  LTC_FN_COUNTER("raf::");
  auto ratex_tensors = LazyTensor::split(bridge::raf_backend::GetLtcTensor(self), split_size, dim);
  return bridge::AtenFromLtcTensors(ratex_tensors);
}

std::vector<at::Tensor> LazyNativeFunctions::split_with_sizes(const at::Tensor& self,
                                                              at::IntArrayRef split_sizes,
                                                              int64_t dim) {
  LTC_FN_COUNTER("raf::");
  auto ratex_tensors = LazyTensor::split_with_sizes(bridge::raf_backend::GetLtcTensor(self),
                                                    Helpers::I64List(split_sizes), dim);
  return bridge::AtenFromLtcTensors(ratex_tensors);
}

at::Tensor LazyNativeFunctions::sqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sqrt(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::squeeze(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::squeeze(bridge::raf_backend::GetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  std::vector<at::Tensor> ratextens_tensors = {self};
  auto ratextens = bridge::LtcCreateTensorList(ratextens_tensors);
  ratextens[0].squeeze_();
  std::vector<size_t> ratextens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(ratextens_tensors, ratextens, ratextens_update_indices);
  } else {
    bridge::LtcUpdateTensors(ratextens_tensors, ratextens, ratextens_update_indices);
  }
  return self;
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  std::vector<at::Tensor> ratextens_tensors = {self};
  auto ratextens = bridge::LtcCreateTensorList(ratextens_tensors);
  ratextens[0].squeeze_(dim);
  std::vector<size_t> ratextens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(ratextens_tensors, ratextens, ratextens_update_indices);
  } else {
    bridge::LtcUpdateTensors(ratextens_tensors, ratextens, ratextens_update_indices);
  }
  return self;
}

at::Tensor LazyNativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::stack(bridge::raf_backend::GetLtcTensors(tensors), dim));
}

at::Tensor LazyNativeFunctions::std(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, /*correction=*/unbiased ? 1 : 0));
}

at::Tensor LazyNativeFunctions::std(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
                                    bool keepdim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      bridge::raf_backend::GetLtcTensor(self), lazy_tensors::util::ToVector<int64_t>(dim), keepdim,
      /*correction=*/unbiased ? 1 : 0));
}

at::Tensor LazyNativeFunctions::std(const at::Tensor& self, at::OptionalIntArrayRef dim,
                                    c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::std(self_tensor,
                      dim ? lazy_tensors::util::ToVector<int64_t>(*dim)
                          : lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
                      keepdim, correction ? *correction : 1));
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self, const at::Tensor& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::sub(xself, xother, alpha, dtype);
                    });
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self, const at::Scalar& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("raf::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::sub(xself, other, alpha, dtype);
                    });
}

at::Tensor LazyNativeFunctions::sum(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::sum(
      self_tensor, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor LazyNativeFunctions::sum(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                    c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::sum(bridge::raf_backend::GetLtcTensor(self),
                                                   lazy_tensors::util::ToVector<int64_t>(dim),
                                                   keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LazyNativeFunctions::svd(const at::Tensor& self,
                                                                        bool some,
                                                                        bool compute_uv) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::svd(bridge::raf_backend::GetLtcTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)),
                         bridge::AtenFromLtcTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::symeig(const at::Tensor& self,
                                                               bool eigenvectors, bool upper) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::symeig(bridge::raf_backend::GetLtcTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::raf_backend::GetLtcTensor(self), 0, 1));
}

at::Tensor& LazyNativeFunctions::t_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("raf::");
    LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
    LazyTensor::transpose_(self_tensor, 0, 1);
    return self;
  }
  return FALLBACK_ATEN_OP(t_, self);
}

at::Tensor LazyNativeFunctions::take(const at::Tensor& self, const at::Tensor& index) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::take(bridge::raf_backend::GetLtcTensor(self),
                                                    bridge::raf_backend::GetLtcTensor(index)));
}

at::Tensor LazyNativeFunctions::tan(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::tan(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::tanh(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::tanh(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::tanh_backward(const at::Tensor& grad_output,
                                              const at::Tensor& output) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::tanh_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(output)));
}

at::Tensor LazyNativeFunctions::threshold(const at::Tensor& self, const at::Scalar& threshold,
                                          const at::Scalar& value) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold(
      bridge::raf_backend::GetLtcTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor LazyNativeFunctions::threshold_backward(const at::Tensor& grad_output,
                                                   const at::Tensor& self,
                                                   const at::Scalar& threshold) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold_backward(
      bridge::raf_backend::GetLtcTensor(grad_output), bridge::raf_backend::GetLtcTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::topk(const at::Tensor& self, int64_t k,
                                                             int64_t dim, bool largest,
                                                             bool sorted) {
  LTC_FN_COUNTER("raf::");
  auto results = LazyTensor::topk(bridge::raf_backend::GetLtcTensor(self), k, dim, largest, sorted);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor LazyNativeFunctions::trace(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::trace(bridge::raf_backend::GetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::raf_backend::GetLtcTensor(self), dim0, dim1));
}

at::Tensor& LazyNativeFunctions::transpose_(at::Tensor& self, int64_t dim0, int64_t dim1) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::triangular_solve(const at::Tensor& b,
                                                                         const at::Tensor& A,
                                                                         bool upper, bool transpose,
                                                                         bool unitriangular) {
  LTC_FN_COUNTER("raf::");
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = LazyTensor::triangular_solve(bridge::raf_backend::GetLtcTensor(b),
                                              bridge::raf_backend::GetLtcTensor(A),
                                              /*left_side=*/true, upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor LazyNativeFunctions::tril(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::tril(bridge::raf_backend::GetLtcTensor(self), diagonal));
}

at::Tensor LazyNativeFunctions::triu(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::triu(bridge::raf_backend::GetLtcTensor(self), diagonal));
}

at::Tensor LazyNativeFunctions::trunc(const at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::trunc(bridge::raf_backend::GetLtcTensor(self)));
}

std::vector<at::Tensor> LazyNativeFunctions::unbind(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensors(
      LazyTensor::unbind(bridge::raf_backend::GetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::uniform_(at::Tensor& self, double from, double to,
                                          c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("raf::");
  if (generator.has_value() && generator->defined()) {
    return FALLBACK_ATEN_OP(uniform_, self, from, to, generator);
  }
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::unsqueeze(bridge::raf_backend::GetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::upsample_bilinear2d(const at::Tensor& self,
                                                    at::IntArrayRef output_size, bool align_corners,
                                                    c10::optional<double> scales_h,
                                                    c10::optional<double> scales_w) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return FALLBACK_ATEN_OP(upsample_bilinear2d, self, output_size, align_corners, scales_h,
                            scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d(
      self_tensor, lazy_tensors::util::ToVector<int64_t>(output_size), align_corners));
}

at::Tensor LazyNativeFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size,
    bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  LTC_FN_COUNTER("raf::");
  LazyTensor grad_output_tensor = bridge::raf_backend::GetLtcTensor(grad_output);
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return FALLBACK_ATEN_OP(upsample_bilinear2d_backward, grad_output, output_size, input_size,
                            align_corners, scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d_backward(
      grad_output_tensor, lazy_tensors::util::ToVector<int64_t>(output_size),
      lazy_tensors::util::ToVector<int64_t>(input_size), align_corners));
}

at::Tensor LazyNativeFunctions::upsample_nearest2d(
    const at::Tensor& input, at::OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("raf::");
  LazyTensor input_tensor = bridge::raf_backend::GetLtcTensor(input);
  if (input_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenRAFTypeDefault::upsample_nearest2d(input, output_size, scale_factors);
  }
  absl::Span<const int64_t> input_dims = input_tensor.shape().get().dimensions();
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      input_tensor, GetOutputSizeWithScale(input_dims, scale_factors, output_size)));
}

at::Tensor LazyNativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::OptionalIntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("raf::");
  LazyTensor grad_output_tensor = bridge::raf_backend::GetLtcTensor(grad_output);
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenRAFTypeDefault::upsample_nearest2d_backward(grad_output, output_size, input_size,
                                                           scale_factors);
  }
  std::vector<int64_t> input_dim = lazy_tensors::util::ToVector<int64_t>(input_size);
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor, GetOutputSizeWithScale(input_dim, scale_factors, output_size),
      input_dim));
}

at::Tensor LazyNativeFunctions::upsample_nearest2d(const at::Tensor& self,
                                                   at::IntArrayRef output_size,
                                                   c10::optional<double> scales_h,
                                                   c10::optional<double> scales_w) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return FALLBACK_ATEN_OP(upsample_nearest2d, self, output_size, scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      self_tensor, lazy_tensors::util::ToVector<int64_t>(output_size)));
}

at::Tensor LazyNativeFunctions::upsample_nearest2d_backward(const at::Tensor& grad_output,
                                                            at::IntArrayRef output_size,
                                                            at::IntArrayRef input_size,
                                                            c10::optional<double> scales_h,
                                                            c10::optional<double> scales_w) {
  LTC_FN_COUNTER("raf::");
  LazyTensor grad_output_tensor = bridge::raf_backend::GetLtcTensor(grad_output);
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return FALLBACK_ATEN_OP(upsample_nearest2d_backward, grad_output, output_size, input_size,
                            scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor, lazy_tensors::util::ToVector<int64_t>(output_size),
      lazy_tensors::util::ToVector<int64_t>(input_size)));
}

at::Tensor LazyNativeFunctions::var(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(bridge::raf_backend::GetLtcTensor(self),
                      lazy_tensors::util::Iota<int64_t>(
                          bridge::raf_backend::GetLtcTensor(self).shape().get().rank()),
                      /*correction=*/unbiased ? 1 : 0,
                      /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::var(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
                                    bool keepdim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::var(self_tensor, Helpers::I64List(dim),
                                                   /*correction=*/unbiased ? 1 : 0, keepdim));
}

at::Tensor LazyNativeFunctions::var(const at::Tensor& self, at::OptionalIntArrayRef dim,
                                    c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(self_tensor,
                      dim ? Helpers::I64List(*dim)
                          : lazy_tensors::util::Iota<int64_t>(
                                bridge::raf_backend::GetLtcTensor(self).shape().get().rank()),
                      correction ? *correction : 1, keepdim));
}

at::Tensor LazyNativeFunctions::view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("raf::");
  return bridge::AtenFromLtcTensor(LazyTensor::view(self_tensor, Helpers::I64List(size)));
}

at::Tensor& LazyNativeFunctions::zero_(at::Tensor& self) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  LazyTensor::zero_(self_tensor);
  return self;
}

at::Scalar LazyNativeFunctions::_local_scalar_dense(const at::Tensor& self) {
  return FALLBACK_ATEN_OP(_local_scalar_dense, self);
}

}  // namespace torch_lazy_tensors
