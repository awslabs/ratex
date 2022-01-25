#include "torch_mnm/csrc/aten_mnm_type.h"

#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>

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

#include "torch_mnm/csrc/ops/log_softmax_backward_use_in.h"
#include "torch_mnm/csrc/aten_mnm_type_default.h"
#include "torch_mnm/csrc/aten_autograd_ops.h"
#include "torch_mnm/csrc/version.h"
#include "torch_mnm/csrc/aten_mnm_bridge.h"
#include "torch_mnm/csrc/utils/debug.h"
#include "torch_mnm/csrc/utils/torch_mnm_logging.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't exist in AtenMNMType,
//   call at::native::func instead.
//   E.g. don't call tensor.is_floating_point() or
//   at::is_floating_point(tensor), use at::native::is_floating_point(tensor).

namespace torch_lazy_tensors {

bool IsSupportedAdaptiveAvgPool(absl::Span<const lazy_tensors::int64> input_size,
                                absl::Span<const lazy_tensors::int64> output_size, int pool_dim) {
  lazy_tensors::int64 rank = input_size.size();
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
  auto xla_device_opt = bridge::GetLtcDevice(device);
  return xla_device_opt ? *xla_device_opt : GetCurrentDevice();
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
      return static_cast<int64_t>(1) << std::numeric_limits<lazy_tensors::half>::digits;
    case lazy_tensors::PrimitiveType::BF16:
      return static_cast<int64_t>(1) << std::numeric_limits<lazy_tensors::bfloat16>::digits;
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
  auto self_xtensor = bridge::mnm_backend::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::mnm_backend::GetLtcTensor(other);
    self_tensor = bridge::GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

// The input is in format of {N, C, H, W} and the output will be {H, W}.
std::vector<lazy_tensors::int64> GetOutputSizeWithScale(
    absl::Span<const lazy_tensors::int64> input_size,
    const c10::optional<at::ArrayRef<double>>& scale_factors,
    const c10::optional<at::IntArrayRef>& output_size) {
  if (!output_size) {
    LTC_CHECK(scale_factors);
    LTC_CHECK_EQ(scale_factors->size(), 2);
    // Calculate the output size from input_shape and scale_factors
    LTC_CHECK_EQ(input_size.size(), 4);
    lazy_tensors::int64 output_h = input_size[2] * (*scale_factors)[0];
    lazy_tensors::int64 output_w = input_size[3] * (*scale_factors)[1];
    return {output_h, output_w};
  }
  LTC_CHECK(!scale_factors);
  return lazy_tensors::util::ToVector<lazy_tensors::int64>(*output_size);
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
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
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

void AtenInitialize() {
  TORCH_MNM_VLOG(1) << "PyTorch GIT revision: " << torch_mnm::TORCH_GITREV;
  TORCH_MNM_VLOG(1) << "MNM GIT revision: " << torch_mnm::MNM_GITREV;

  LTCTensorImpl::AtenInitialize();
}

}  // namespace

at::Tensor& AtenMNMType::__ilshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenMNMType::__ilshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
  return self;
}

at::Tensor& AtenMNMType::__irshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenMNMType::__irshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
  return self;
}

at::Tensor AtenMNMType::__lshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor AtenMNMType::__lshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenMNMType::__rshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor AtenMNMType::__rshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenMNMType::_adaptive_avg_pool3d(const at::Tensor& self, at::IntArrayRef output_size) {
  LTC_FN_COUNTER("mnm::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size_list,
                                  /*pool_dim=*/3)) {
    return AtenMNMTypeDefault::_adaptive_avg_pool3d(self, output_size);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::adaptive_avg_pool3d(bridge::mnm_backend::GetLtcTensor(self), output_size_list));
}

at::Tensor AtenMNMType::_adaptive_avg_pool3d_backward(const at::Tensor& grad_output,
                                                      const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  int64_t rank = grad_output.dim();
  std::vector<lazy_tensors::int64> output_size{
      grad_output.size(rank - 3), grad_output.size(rank - 2), grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/3)) {
    return AtenMNMTypeDefault::_adaptive_avg_pool3d_backward(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
  LTC_FN_COUNTER("mnm::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size_list,
                                  /*pool_dim=*/2)) {
    return AtenMNMTypeDefault::_adaptive_avg_pool2d(self, output_size);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::_adaptive_avg_pool2d(bridge::mnm_backend::GetLtcTensor(self), output_size_list));
}

at::Tensor AtenMNMType::_adaptive_avg_pool2d_backward(const at::Tensor& grad_output,
                                                      const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  int64_t rank = grad_output.dim();
  std::vector<lazy_tensors::int64> output_size{grad_output.size(rank - 2),
                                               grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/2)) {
    return AtenMNMTypeDefault::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self)));
}

void AtenMNMType::_amp_foreach_non_finite_check_and_unscale_(at::TensorList self,
                                                             at::Tensor& found_inf,
                                                             const at::Tensor& inv_scale) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor found_inf_tensor = bridge::mnm_backend::GetLtcTensor(found_inf);
  DeviceType hw_type = found_inf_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  LazyTensor::_amp_foreach_non_finite_check_and_unscale_(
      bridge::mnm_backend::GetLtcTensors(self), found_inf_tensor,
      bridge::mnm_backend::GetLtcTensor(inv_scale));
}

at::Tensor& AtenMNMType::_amp_update_scale_(at::Tensor& current_scale, at::Tensor& growth_tracker,
                                            const at::Tensor& found_inf, double scale_growth_factor,
                                            double scale_backoff_factor, int64_t growth_interval) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor growth_tracker_tensor = bridge::mnm_backend::GetLtcTensor(growth_tracker);
  LazyTensor current_scale_tensor = bridge::mnm_backend::GetLtcTensor(current_scale);
  DeviceType hw_type = growth_tracker_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  LazyTensor::_amp_update_scale_(growth_tracker_tensor, current_scale_tensor,
                                 bridge::mnm_backend::GetLtcTensor(found_inf), scale_growth_factor,
                                 scale_backoff_factor, growth_interval);
  return current_scale;
}

at::Tensor AtenMNMType::_copy_from(const at::Tensor& self, const at::Tensor& dst,
                                   bool non_blocking) {
  LTC_FN_COUNTER("mnm::");
  auto dst_tensor = bridge::mnm_backend::TryGetLtcTensor(dst);
  auto self_tensor = bridge::mnm_backend::TryGetLtcTensor(self);
  if (!self_tensor) {
    static bool sync_update = lazy_tensors::sys_util::GetEnvBool("MNM_TENSOR_UPDATE_SYNC", true);
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor = CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    if (!dst_tensor->CurrentIrValue()) {
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      LTC_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/true));
      }
    } else {
      LazyTensor::copy_(*dst_tensor, *self_tensor);
      bridge::ReplaceLtcTensor(dst, *dst_tensor);
    }
  }
  return dst;
}

at::Tensor& AtenMNMType::_index_put_impl_(at::Tensor& self,
                                          const c10::List<c10::optional<at::Tensor>>& indices,
                                          const at::Tensor& values, bool accumulate,
                                          bool /* unsafe */) {
  LTC_FN_COUNTER("mnm::");
  return index_put_(self, indices, values, accumulate);
}

at::Tensor AtenMNMType::_log_softmax(const at::Tensor& self, int64_t dim,
                                     bool /* half_to_float */) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::log_softmax(bridge::mnm_backend::GetLtcTensor(self), dim, c10::nullopt));
}

ir::Value MaybeCastIrValue(const LazyTensor& self, ir::Value ir_value, const Device& device,
                           c10::optional<at::ScalarType> logical_element_type) {
  if (!logical_element_type) {
    logical_element_type = self.dtype_optional();
  }
  if (logical_element_type && RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = ir::MakeNode<ir::ops::Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

LazyTensor CreateFrom(const LazyTensor& self, ir::Value ir_value) {
  ir_value = MaybeCastIrValue(self, std::move(ir_value), self.GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return LazyTensor::Create(std::move(ir_value), self.GetDevice(), self.dtype_optional());
}

ir::NodePtr LogSoftmaxBackwardUseInOp(const ir::Value& grad_output, const ir::Value& output,
                                      lazy_tensors::int64 dim, const ir::Value& self) {
  return ir::MakeNode<ir::ops::LogSoftmaxBackwardUseIn>(
      grad_output, output, Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()),
      self);
}

LazyTensor log_softmax_backward(const LazyTensor& grad_output, const LazyTensor& output,
                                lazy_tensors::int64 dim, const LazyTensor& self) {
  return CreateFrom(grad_output,
                    LogSoftmaxBackwardUseInOp(grad_output.GetIrValue(), output.GetIrValue(), dim,
                                              self.GetIrValue()));
}

at::Tensor AtenMNMType::_log_softmax_backward_data(const at::Tensor& grad_output,
                                                   const at::Tensor& output, int64_t dim,
                                                   const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(log_softmax_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(output),
      dim, bridge::mnm_backend::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::_pack_padded_sequence(const at::Tensor& input,
                                                                      const at::Tensor& lengths,
                                                                      bool batch_first) {
  LTC_FN_COUNTER("aten::");
  std::vector<at::Tensor> xla_tensors = {lengths};
  auto cpu_tensors = bridge::LtcCreateTensorList(xla_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor AtenMNMType::_s_where(const at::Tensor& condition, const at::Tensor& self,
                                 const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::where(bridge::mnm_backend::GetLtcTensor(condition),
                                                     bridge::mnm_backend::GetLtcTensor(self),
                                                     bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor AtenMNMType::_softmax(const at::Tensor& self, int64_t dim, bool /* half_to_float */) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax(bridge::mnm_backend::GetLtcTensor(self), dim, c10::nullopt));
}

at::Tensor AtenMNMType::_softmax_backward_data(const at::Tensor& grad_output,
                                               const at::Tensor& output, int64_t dim,
                                               const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax_backward(bridge::mnm_backend::GetLtcTensor(grad_output),
                                   bridge::mnm_backend::GetLtcTensor(output), dim));
}

at::Tensor AtenMNMType::_trilinear(const at::Tensor& i1, const at::Tensor& i2, const at::Tensor& i3,
                                   at::IntArrayRef expand1, at::IntArrayRef expand2,
                                   at::IntArrayRef expand3, at::IntArrayRef sumdim,
                                   int64_t unroll_dim) {
  return AtenMNMTypeDefault::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

at::Tensor AtenMNMType::_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("mnm::");
  return view(self, size);
}

at::Tensor AtenMNMType::abs(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::abs(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::abs_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::abs_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::acos(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::acos(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::acos_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::acos_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::acosh(const at::Tensor& self) {
  return AtenMNMTypeDefault::acosh(self);
}

at::Tensor AtenMNMType::add(const at::Tensor& self, const at::Tensor& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenMNMType::add(const at::Tensor& self, const at::Scalar& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::add(xself, other, alpha, dtype);
                    });
}

at::Tensor& AtenMNMType::add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::add_(self_tensor, bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()),
                   alpha);
  return self;
}

at::Tensor& AtenMNMType::add_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::add_(self_tensor, other, alpha);
    return self;
  }
  return AtenMNMTypeDefault::add_(self, other, alpha);
}

at::Tensor AtenMNMType::addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                                const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcdiv(
      bridge::mnm_backend::GetLtcTensor(self), value, bridge::mnm_backend::GetLtcTensor(tensor1),
      bridge::mnm_backend::GetLtcTensor(tensor2)));
}

at::Tensor& AtenMNMType::addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2, const at::Scalar& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::addcdiv_(self_tensor, value, bridge::mnm_backend::GetLtcTensor(tensor1),
                         bridge::mnm_backend::GetLtcTensor(tensor2));
    return self;
  }
  return AtenMNMTypeDefault::addcdiv_(self, tensor1, tensor2, value);
}

at::Tensor AtenMNMType::addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                                const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcmul(
      bridge::mnm_backend::GetLtcTensor(self), value, bridge::mnm_backend::GetLtcTensor(tensor1),
      bridge::mnm_backend::GetLtcTensor(tensor2)));
}

at::Tensor& AtenMNMType::addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2, const at::Scalar& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::addcmul_(self_tensor, value, bridge::mnm_backend::GetLtcTensor(tensor1),
                         bridge::mnm_backend::GetLtcTensor(tensor2));
    return self;
  }
  return AtenMNMTypeDefault::addcmul_(self, tensor1, tensor2, value);
}

at::Tensor AtenMNMType::addmm(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, const at::Scalar& beta,
                              const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (beta.to<double>() != 1 || alpha.to<double>() != 1 || !at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat1) || !at::native::is_floating_point(mat2)) {
    return AtenMNMTypeDefault::addmm(self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::addmm(bridge::mnm_backend::GetLtcTensor(mat1),
                        /*weight=*/bridge::mnm_backend::GetLtcTensor(mat2),
                        /*bias=*/bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return self;
}

at::Tensor AtenMNMType::all(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::all(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenMNMType::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::all(bridge::mnm_backend::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor AtenMNMType::any(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::any(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenMNMType::any(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::any(bridge::mnm_backend::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor& AtenMNMType::arange_out(const at::Scalar& start, const at::Scalar& end,
                                    const at::Scalar& step, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor AtenMNMType::argmax(const at::Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  return dim ? bridge::AtenFromLtcTensor(
                   LazyTensor::argmax(bridge::mnm_backend::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmax(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::argmin(const at::Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  return dim ? bridge::AtenFromLtcTensor(
                   LazyTensor::argmin(bridge::mnm_backend::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmin(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                   at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(self_tensor.shape(), xsize, xstride,
                                             storage_offset.value_or(0))) {
    return AtenMNMTypeDefault::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::as_strided(
      self_tensor, std::move(xsize), std::move(xstride), Helpers::I64Optional(storage_offset)));
}

const at::Tensor& AtenMNMType::as_strided_(const at::Tensor& self, at::IntArrayRef size,
                                           at::IntArrayRef stride,
                                           c10::optional<int64_t> storage_offset) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (ir::ops::AsStrided::StrideIsSupported(self_tensor.shape(), xsize, xstride,
                                            storage_offset.value_or(0))) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                            Helpers::I64Optional(storage_offset));
    return self;
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::as_strided_", 1);
  TORCH_MNM_VLOG(3) << "XLA as_strided_ :"
                    << " self=" << self.toString();
  auto xlatens = bridge::LtcCreateTensorList({self});
  at::as_strided_(xlatens[0], size, stride, storage_offset);
  bridge::LtcUpdateTensors({self}, xlatens, {0});
  return self;
}

at::Tensor AtenMNMType::asin(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::asin(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::asin_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::asin_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::asin_(self);
}

at::Tensor AtenMNMType::asinh(const at::Tensor& self) {
  return AtenMNMTypeDefault::asinh(self);
}

at::Tensor& AtenMNMType::asinh_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::asinh_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::atan(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::atan(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::atanh(const at::Tensor& self) {
  return AtenMNMTypeDefault::atanh(self);
}

at::Tensor AtenMNMType::atan2(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return AtenMNMTypeDefault::atan2(self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor& AtenMNMType::atan2_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  // xla::Atan2 doesn't support integer types.
  if (!bridge::IsInteropView(self) && self.is_floating_point() && other.is_floating_point()) {
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::atan2_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::atan2_(self, other);
}

at::Tensor& AtenMNMType::atan_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::atan_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::atan_(self);
}

at::Tensor& AtenMNMType::atanh_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::atanh_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad,
                                   c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("mnm::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenMNMTypeDefault::avg_pool2d(self, kernel_size, stride, padding, ceil_mode,
                                          count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::avg_pool_nd(bridge::mnm_backend::GetLtcTensor(self), /*spatial_dim_count=*/2,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenMNMType::avg_pool2d_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                            at::IntArrayRef padding, bool ceil_mode,
                                            bool count_include_pad,
                                            c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("mnm::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenMNMTypeDefault::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding,
                                                   ceil_mode, count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenMNMType::avg_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad,
                                   c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("mnm::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenMNMTypeDefault::avg_pool3d(self, kernel_size, stride, padding, ceil_mode,
                                          count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::avg_pool_nd(bridge::mnm_backend::GetLtcTensor(self), /*spatial_dim_count=*/3,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenMNMType::avg_pool3d_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                            at::IntArrayRef padding, bool ceil_mode,
                                            bool count_include_pad,
                                            c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("mnm::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenMNMTypeDefault::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding,
                                                   ceil_mode, count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenMNMType::baddbmm(const at::Tensor& self, const at::Tensor& batch1,
                                const at::Tensor& batch2, const at::Scalar& beta,
                                const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) || !at::native::is_floating_point(batch2)) {
    return AtenMNMTypeDefault::baddbmm(self, batch1, batch2, beta, alpha);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::baddbmm(
      bridge::mnm_backend::GetLtcTensor(self), bridge::mnm_backend::GetLtcTensor(batch1),
      bridge::mnm_backend::GetLtcTensor(batch2), beta, alpha));
}

at::Tensor& AtenMNMType::baddbmm_(at::Tensor& self, const at::Tensor& batch1,
                                  const at::Tensor& batch2, const at::Scalar& beta,
                                  const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) || !at::native::is_floating_point(batch2)) {
    return AtenMNMTypeDefault::baddbmm_(self, batch1, batch2, beta, alpha);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::baddbmm_(self_tensor, bridge::mnm_backend::GetLtcTensor(batch1),
                       bridge::mnm_backend::GetLtcTensor(batch2), beta, alpha);
  return self;
}

at::Tensor AtenMNMType::bernoulli(const at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::bernoulli(self, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::bernoulli(self_tensor));
}

at::Tensor& AtenMNMType::bernoulli_(at::Tensor& self, double p,
                                    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::bernoulli_(self, p, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& AtenMNMType::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::bernoulli_(self, p, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, bridge::mnm_backend::GetLtcTensor(p));
  return self;
}

at::Tensor AtenMNMType::binary_cross_entropy(const at::Tensor& self, const at::Tensor& target,
                                             const c10::optional<at::Tensor>& weight,
                                             int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy(
      self_tensor, bridge::mnm_backend::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor AtenMNMType::binary_cross_entropy_backward(const at::Tensor& grad_output,
                                                      const at::Tensor& self,
                                                      const at::Tensor& target,
                                                      const c10::optional<at::Tensor>& weight,
                                                      int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), self_tensor,
      bridge::mnm_backend::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor AtenMNMType::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight, int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor& AtenMNMType::bitwise_and_out(const at::Tensor& self, const at::Scalar& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_and_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& AtenMNMType::bitwise_and_out(const at::Tensor& self, const at::Tensor& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_and_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self),
                              bridge::mnm_backend::GetLtcTensor(other));
  return out;
}

at::Tensor& AtenMNMType::bitwise_not_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::bitwise_not_out(out_tensor, self_tensor);
  return out;
}

at::Tensor& AtenMNMType::bitwise_or_out(const at::Tensor& self, const at::Scalar& other,
                                        at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& AtenMNMType::bitwise_or_out(const at::Tensor& self, const at::Tensor& other,
                                        at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self),
                             bridge::mnm_backend::GetLtcTensor(other));
  return out;
}

at::Tensor& AtenMNMType::bitwise_xor_out(const at::Tensor& self, const at::Scalar& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self), other);
  return out;
}

at::Tensor& AtenMNMType::bitwise_xor_out(const at::Tensor& self, const at::Tensor& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self),
                              bridge::mnm_backend::GetLtcTensor(other));
  return out;
}

at::Tensor AtenMNMType::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(mat2)) {
    return AtenMNMTypeDefault::bmm(self, mat2);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::bmm(bridge::mnm_backend::GetLtcTensor(self),
                                                   bridge::mnm_backend::GetLtcTensor(mat2)));
}

at::Tensor AtenMNMType::cat(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cat(bridge::mnm_backend::GetLtcTensors(tensors), dim));
}

at::Tensor AtenMNMType::ceil(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ceil(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::ceil_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::ceil_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::ceil_(self);
}

at::Tensor AtenMNMType::cholesky(const at::Tensor& self, bool upper) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cholesky(bridge::mnm_backend::GetLtcTensor(self), upper));
}

at::Tensor AtenMNMType::clamp(const at::Tensor& self, const c10::optional<at::Scalar>& min,
                              const c10::optional<at::Scalar>& max) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::mnm_backend::GetLtcTensor(self), min, max));
}

at::Tensor AtenMNMType::clamp(const at::Tensor& self, const c10::optional<at::Tensor>& min,
                              const c10::optional<at::Tensor>& max) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::mnm_backend::GetLtcTensor(self), min, max));
}

at::Tensor& AtenMNMType::clamp_(at::Tensor& self, const c10::optional<at::Scalar>& min,
                                const c10::optional<at::Scalar>& max) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min, max);
    return self;
  }
  return AtenMNMTypeDefault::clamp_(self, min, max);
}

at::Tensor AtenMNMType::clamp_max(const at::Tensor& self, const at::Scalar& max) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::mnm_backend::GetLtcTensor(self), c10::nullopt, max));
}

at::Tensor& AtenMNMType::clamp_max_(at::Tensor& self, const at::Scalar& max) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, c10::nullopt, max);
    return self;
  }
  return AtenMNMTypeDefault::clamp_max_(self, max);
}

at::Tensor& AtenMNMType::clamp_max_out(const at::Tensor& self, const at::Tensor& max,
                                       at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self), c10::nullopt, max);
  return out;
}

at::Tensor AtenMNMType::clamp_min(const at::Tensor& self, const at::Scalar& min) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::mnm_backend::GetLtcTensor(self), min, c10::nullopt));
}

at::Tensor& AtenMNMType::clamp_min_(at::Tensor& self, const at::Scalar& min) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min, c10::nullopt);
    return self;
  }
  return AtenMNMTypeDefault::clamp_min_(self, min);
}

at::Tensor& AtenMNMType::clamp_min_out(const at::Tensor& self, const at::Tensor& min,
                                       at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self), min, c10::nullopt);
  return out;
}

at::Tensor AtenMNMType::clone(const at::Tensor& self,
                              c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor));
}

at::Tensor AtenMNMType::constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad,
                                        const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::constant_pad_nd(
      bridge::mnm_backend::GetLtcTensor(self), Helpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor AtenMNMType::convolution_overrideable(const at::Tensor& input, const at::Tensor& weight,
                                                 const c10::optional<at::Tensor>& bias,
                                                 at::IntArrayRef stride, at::IntArrayRef padding,
                                                 at::IntArrayRef dilation, bool transposed,
                                                 at::IntArrayRef output_padding, int64_t groups) {
  LTC_FN_COUNTER("mnm::");
  if (IsDefined(bias)) {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::mnm_backend::GetLtcTensor(input), bridge::mnm_backend::GetLtcTensor(weight),
        bridge::mnm_backend::GetLtcTensor(*bias), Helpers::I64List(stride),
        Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  } else {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::mnm_backend::GetLtcTensor(input), bridge::mnm_backend::GetLtcTensor(weight),
        Helpers::I64List(stride), Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  }
}

// This functions covers the whole convolution backward lowering.
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenMNMType::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("mnm::");
  auto gradients = LazyTensor::convolution_backward_overrideable(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(input),
      bridge::mnm_backend::GetLtcTensor(weight), Helpers::I64List(stride),
      Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
      Helpers::I64List(output_padding), groups);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients)) : at::Tensor(),
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients)) : at::Tensor(),
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients)) : at::Tensor());
}

at::Tensor AtenMNMType::cos(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::cos(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::cos_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::cos_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::cos_(self);
}

at::Tensor AtenMNMType::cosh(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::cosh(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::cosh_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::cosh_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::cosh_(self);
}

at::Tensor AtenMNMType::cross(const at::Tensor& self, const at::Tensor& other,
                              c10::optional<int64_t> dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::cross(bridge::mnm_backend::GetLtcTensor(self),
                                                     bridge::mnm_backend::GetLtcTensor(other),
                                                     Helpers::I64Optional(dim)));
}

at::Tensor AtenMNMType::cumprod(const at::Tensor& self, int64_t dim,
                                c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  c10::optional<at::ScalarType> promoted_dtype = PromoteIntegralType(self_tensor.dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return AtenMNMTypeDefault::cumprod(self, dim, dtype);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor AtenMNMType::cumsum(const at::Tensor& self, int64_t dim,
                               c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  if (IsOperationOnType(dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return AtenMNMTypeDefault::cumsum(self, dim, dtype);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::cumsum(self_tensor, dim, dtype));
}

at::Tensor AtenMNMType::diag(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diag(bridge::mnm_backend::GetLtcTensor(self), diagonal));
}

at::Tensor AtenMNMType::diagonal(const at::Tensor& self, int64_t offset, int64_t dim1,
                                 int64_t dim2) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diagonal(bridge::mnm_backend::GetLtcTensor(self), offset, dim1, dim2));
}

at::Tensor AtenMNMType::div(const at::Tensor& self, const at::Tensor& other) {
  return div(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor AtenMNMType::div(const at::Tensor& self, const at::Tensor& other,
                            c10::optional<std::string> rounding_mode) {
  LTC_FN_COUNTER("mnm::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, other);
  return bridge::AtenFromLtcTensor(
      LazyTensor::div(operands.first, operands.second, rounding_mode, dtype));
}

at::Tensor AtenMNMType::div(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::div(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor& AtenMNMType::div_(at::Tensor& self, const at::Tensor& other) {
  return div_(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor& AtenMNMType::div_(at::Tensor& self, const at::Tensor& other,
                              c10::optional<std::string> rounding_mode) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::div_(self_tensor, bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()),
                     rounding_mode);
    return self;
  }
  return AtenMNMTypeDefault::div_(self, other, rounding_mode);
}

at::Tensor& AtenMNMType::div_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::div_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::div_(self, other);
}

at::Tensor AtenMNMType::dot(const at::Tensor& self, const at::Tensor& tensor) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK_EQ(self.dim(), 1) << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  LTC_CHECK_EQ(tensor.dim(), 1) << "dot: Expected 1-D argument tensor, but got " << tensor.dim()
                                << "-D";
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(tensor)) {
    return AtenMNMTypeDefault::dot(self, tensor);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::matmul(bridge::mnm_backend::GetLtcTensor(self),
                                                      bridge::mnm_backend::GetLtcTensor(tensor)));
}

at::Tensor AtenMNMType::elu(const at::Tensor& self, const at::Scalar& alpha,
                            const at::Scalar& scale, const at::Scalar& input_scale) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu(bridge::mnm_backend::GetLtcTensor(self), alpha, scale, input_scale));
}

at::Tensor& AtenMNMType::elu_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                              const at::Scalar& input_scale) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::elu_(self_tensor, alpha, scale, input_scale);
    return self;
  }
  return AtenMNMTypeDefault::elu_(self, alpha, scale, input_scale);
}

at::Tensor AtenMNMType::elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha,
                                     const at::Scalar& scale, const at::Scalar& input_scale,
                                     bool self, const at::Tensor& self_or_result) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu_backward(bridge::mnm_backend::GetLtcTensor(grad_output), alpha, scale,
                               input_scale, bridge::mnm_backend::GetLtcTensor(self_or_result)));
}

at::Tensor AtenMNMType::embedding(const at::Tensor& weight, const at::Tensor& indices,
                                  int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  LTC_FN_COUNTER("mnm::");
  // TODO: for now route to native, which dispatches supported XLA operations.
  // We need to make use of the TPU embedding core here eventually.
  return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor AtenMNMType::embedding_dense_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& indices, int64_t num_weights,
                                                 int64_t padding_idx, bool scale_grad_by_freq) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::embedding_dense_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor AtenMNMType::empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                              c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                              c10::optional<bool> pin_memory,
                              c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("mnm::");
  // PT empty*() are optimizations to avoid initializing the data when it is
  // known it will be completely rewritten. But since for us doing a zero*()
  // does not actually end up doing any memory initialization, we use that and
  // avoid going to CPU for it. A common PT pattern is indeed doing empty()
  // plus s_copy_().
  return bridge::AtenFromLtcTensor(LazyTensor::full(
      Helpers::I64List(size), 0, GetLtcDeviceOrCurrent(device), GetScalarTypeOrFloat(dtype)));
}

at::Tensor AtenMNMType::empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("mnm::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor AtenMNMType::eq(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::eq(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::eq(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::eq(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::eq_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::eq_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::eq_(self, other);
}

at::Tensor& AtenMNMType::eq_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::eq_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::eq_(self, other);
}

at::Tensor AtenMNMType::erf(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::erf(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::erf_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::erf_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::erf_(self);
}

at::Tensor AtenMNMType::erfc(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::erfc(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::erfc_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::erfc_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::erfc_(self);
}

at::Tensor AtenMNMType::erfinv(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::erfinv(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::erfinv_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::erfinv_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::exp(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::exp(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::exp_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::exp_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::exp_(self);
}

at::Tensor AtenMNMType::expand(const at::Tensor& self, at::IntArrayRef size, bool implicit) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::expand(bridge::mnm_backend::GetLtcTensor(self),
                         lazy_tensors::util::ToVector<lazy_tensors::int64>(size)));
}

at::Tensor AtenMNMType::expm1(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::expm1(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::expm1_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::expm1_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::expm1_(self);
}

at::Tensor& AtenMNMType::exponential_(at::Tensor& self, double lambd,
                                      c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::exponential_(self, lambd, generator);
  }
  LTC_CHECK_GE(lambd, 0.0);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& AtenMNMType::eye_out(int64_t n, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& AtenMNMType::eye_out(int64_t n, int64_t m, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& AtenMNMType::fill_(at::Tensor& self, const at::Scalar& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::fill_(self_tensor, value);
    return self;
  }
  return AtenMNMTypeDefault::fill_(self, value);
}

at::Tensor& AtenMNMType::fill_(at::Tensor& self, const at::Tensor& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LTC_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                                 << "value tensor, but got tensor "
                                 << "with " << value.dim() << " dimension(s).";
    return fill_(self, value.item());
  }
  return AtenMNMTypeDefault::fill_(self, value);
}

at::Tensor AtenMNMType::flip(const at::Tensor& self, at::IntArrayRef dims) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::flip(bridge::mnm_backend::GetLtcTensor(self), Helpers::I64List(dims)));
}

at::Tensor AtenMNMType::floor(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::floor(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::floor_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::floor_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::floor_(self);
}

at::Tensor AtenMNMType::fmod(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor AtenMNMType::fmod(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, other, dtype);
                    });
}

at::Tensor& AtenMNMType::fmod_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::fmod_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::fmod_(self, other);
}

at::Tensor& AtenMNMType::fmod_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::fmod_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::fmod_(self, other);
}

at::Tensor AtenMNMType::frac(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::frac(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::frac_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::frac_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::frac_(self);
}

at::Tensor AtenMNMType::gather(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                               bool /* sparse_grad */) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::gather(bridge::mnm_backend::GetLtcTensor(self), dim,
                                                      bridge::mnm_backend::GetLtcTensor(index)));
}

at::Tensor AtenMNMType::ge(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ge(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::ge(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ge(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::ge_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::ge_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::ge_(self, other);
}

at::Tensor& AtenMNMType::ge_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::ge_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::ge_(self, other);
}

at::Tensor AtenMNMType::gelu(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::gelu_backward(const at::Tensor& grad, const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::mnm_backend::GetLtcTensor(grad), bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::ger(const at::Tensor& self, const at::Tensor& vec2) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ger(bridge::mnm_backend::GetLtcTensor(self),
                                                   bridge::mnm_backend::GetLtcTensor(vec2)));
}

at::Tensor AtenMNMType::gt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::gt(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::gt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::gt(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::gt_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::gt_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::gt_(self, other);
}

at::Tensor& AtenMNMType::gt_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::gt_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::gt_(self, other);
}

at::Tensor AtenMNMType::hardshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardshrink(bridge::mnm_backend::GetLtcTensor(self), lambda));
}

at::Tensor AtenMNMType::hardsigmoid(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardsigmoid(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::hardsigmoid_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::hardsigmoid_(self_tensor);
  return self;
}

at::Tensor AtenMNMType::hardsigmoid_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardsigmoid_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::hardshrink_backward(const at::Tensor& grad_out, const at::Tensor& self,
                                            const at::Scalar& lambda) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardshrink_backward(bridge::mnm_backend::GetLtcTensor(grad_out),
                                      bridge::mnm_backend::GetLtcTensor(self), lambda));
}

at::Tensor AtenMNMType::hardtanh(const at::Tensor& self, const at::Scalar& min_val,
                                 const at::Scalar& max_val) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::mnm_backend::GetLtcTensor(self), min_val, max_val));
}

at::Tensor& AtenMNMType::hardtanh_(at::Tensor& self, const at::Scalar& min_val,
                                   const at::Scalar& max_val) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min_val, max_val);
    return self;
  }
  return AtenMNMTypeDefault::hardtanh_(self, min_val, max_val);
}

at::Tensor AtenMNMType::hardtanh_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                          const at::Scalar& min_val, const at::Scalar& max_val) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardtanh_backward(bridge::mnm_backend::GetLtcTensor(grad_output),
                                    bridge::mnm_backend::GetLtcTensor(self), min_val, max_val));
}

at::Tensor AtenMNMType::index(const at::Tensor& self,
                              const c10::List<c10::optional<at::Tensor>>& indices) {
  LTC_FN_COUNTER("mnm::");
  CanonicalIndexInfo canonical_index_info = GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromLtcTensor(
      LazyTensor::index(bridge::mnm_backend::GetLtcTensor(canonical_index_info.base),
                        bridge::mnm_backend::GetLtcTensors(canonical_index_info.indices),
                        canonical_index_info.start_dim));
}

at::Tensor& AtenMNMType::index_add_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                    const at::Tensor& source) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::index_add_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index),
                         bridge::mnm_backend::GetLtcTensor(source));
  return self;
}

at::Tensor& AtenMNMType::index_copy_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                     const at::Tensor& source) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::index_copy_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index),
                          bridge::mnm_backend::GetLtcTensor(source));
  return self;
}

at::Tensor& AtenMNMType::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                     const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index), value);
  return self;
}

at::Tensor& AtenMNMType::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                     const at::Tensor& value) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index),
                          bridge::mnm_backend::GetLtcTensor(value));
  return self;
}

at::Tensor& AtenMNMType::index_put_(at::Tensor& self,
                                    const c10::List<c10::optional<at::Tensor>>& indices,
                                    const at::Tensor& values, bool accumulate) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK(self.scalar_type() == values.scalar_type());
  CanonicalIndexInfo canonical_index_info = GetCanonicalIndexInfo(self, indices);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::index_put_(self_tensor, bridge::mnm_backend::GetLtcTensor(canonical_index_info.base),
                         bridge::mnm_backend::GetLtcTensors(canonical_index_info.indices),
                         canonical_index_info.start_dim, bridge::mnm_backend::GetLtcTensor(values),
                         accumulate, canonical_index_info.result_permutation);
  return self;
}

at::Tensor AtenMNMType::index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::index_select(
      bridge::mnm_backend::GetLtcTensor(self), dim, bridge::mnm_backend::GetLtcTensor(index)));
}

at::Tensor AtenMNMType::inverse(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::inverse(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::kl_div(const at::Tensor& self, const at::Tensor& target, int64_t reduction,
                               bool log_target) {
  LTC_FN_COUNTER("mnm::");
  return at::native::kl_div(self, target, reduction, log_target);
}

at::Tensor AtenMNMType::kl_div_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                        const at::Tensor& target, int64_t reduction,
                                        bool log_target) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::kl_div_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(target), reduction, log_target));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::kthvalue(const at::Tensor& self, int64_t k,
                                                         int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::kthvalue(bridge::mnm_backend::GetLtcTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor AtenMNMType::l1_loss(const at::Tensor& self, const at::Tensor& target,
                                int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss(bridge::mnm_backend::GetLtcTensor(self),
                                                       bridge::mnm_backend::GetLtcTensor(target),
                                                       reduction));
}

at::Tensor AtenMNMType::l1_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                         const at::Tensor& target, int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(target), reduction));
}

at::Tensor AtenMNMType::le(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::le(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::le(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::le(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::le_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::le_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::le_(self, other);
}

at::Tensor& AtenMNMType::le_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::le_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::le_(self, other);
}

at::Tensor AtenMNMType::leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::leaky_relu(bridge::mnm_backend::GetLtcTensor(self), negative_slope.to<double>()));
}

at::Tensor& AtenMNMType::leaky_relu_(at::Tensor& self, const at::Scalar& negative_slope) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::leaky_relu_(self_tensor, negative_slope.to<double>());
    return self;
  }
  return AtenMNMTypeDefault::leaky_relu_(self, negative_slope);
}

at::Tensor AtenMNMType::leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                            const at::Scalar& negative_slope, bool self_is_result) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK(!self_is_result || negative_slope.to<double>() > 0.0);
  return bridge::AtenFromLtcTensor(LazyTensor::leaky_relu_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      negative_slope.to<double>()));
}

at::Tensor AtenMNMType::log(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::log(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::log10(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(bridge::mnm_backend::GetLtcTensor(self),
                                                        ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor& AtenMNMType::log10_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::log_base_(self_tensor, ir::OpKind(at::aten::log10), 10.0);
  return self;
}

at::Tensor AtenMNMType::log1p(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::log1p(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::log1p_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::log1p_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::log1p_(self);
}

at::Tensor AtenMNMType::log2(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(bridge::mnm_backend::GetLtcTensor(self),
                                                        ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor& AtenMNMType::log2_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::log_base_(self_tensor, ir::OpKind(at::aten::log2), 2.0);
  return self;
}

at::Tensor& AtenMNMType::log_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::log_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::log_(self);
}

at::Tensor AtenMNMType::log_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                             const at::Tensor& buffer) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_sigmoid_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::log_sigmoid_forward(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  auto result_tuple = LazyTensor::log_sigmoid_forward(bridge::mnm_backend::GetLtcTensor(self));
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromLtcTensor(std::get<1>(result_tuple)));
}

at::Tensor AtenMNMType::logsumexp(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  return AtenMNMTypeDefault::logsumexp(self, dim, keepdim);
}

at::Tensor AtenMNMType::logdet(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::logdet(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::lt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::lt(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::lt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::lt(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::masked_fill_(self_tensor, bridge::mnm_backend::GetLtcTensor(mask), value);
  return self;
}

at::Tensor& AtenMNMType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      const at::Tensor& value) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor& AtenMNMType::masked_scatter_(at::Tensor& self, const at::Tensor& mask,
                                         const at::Tensor& source) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::masked_scatter_(self_tensor, bridge::mnm_backend::GetLtcTensor(mask),
                              bridge::mnm_backend::GetLtcTensor(source));
  return self;
}

at::Tensor AtenMNMType::masked_select(const at::Tensor& self, const at::Tensor& mask) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  // Initially make XLA handled masked_select() handling experimental, and
  // opt-in.
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return AtenMNMTypeDefault::masked_select(self, mask);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::masked_select(self_tensor, bridge::mnm_backend::GetLtcTensor(mask)));
}

at::Tensor AtenMNMType::max(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::max(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::lt_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::lt_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::lt_(self, other);
}

at::Tensor& AtenMNMType::lt_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::lt_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::lt_(self, other);
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::max(const at::Tensor& self, int64_t dim,
                                                    bool keepdim) {
  return AtenMNMTypeDefault::max(self, dim, keepdim);
}

at::Tensor AtenMNMType::maximum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::max(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMNMType::max_out(const at::Tensor& self, int64_t dim,
                                                          bool keepdim, at::Tensor& max,
                                                          at::Tensor& max_values) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor max_tensor = bridge::mnm_backend::GetLtcTensor(max);
  LazyTensor max_values_tensor = bridge::mnm_backend::GetLtcTensor(max_values);
  LazyTensor::max_out(max_tensor, max_values_tensor, bridge::mnm_backend::GetLtcTensor(self), dim,
                      keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor AtenMNMType::max_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride, at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("mnm::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(self, kernel_size, stride, padding,
                                                             dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("mnm::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenMNMTypeDefault::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation,
                                                       ceil_mode);
  }
  auto outputs =
      LazyTensor::max_pool_nd(bridge::mnm_backend::GetLtcTensor(self), /*spatial_dim_count=*/2,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor AtenMNMType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("mnm::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenMNMTypeDefault::max_pool2d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));
}

at::Tensor AtenMNMType::max_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride, at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("mnm::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(self, kernel_size, stride, padding,
                                                             dilation, ceil_mode);
}

at::Tensor AtenMNMType::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("mnm::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenMNMTypeDefault::max_pool3d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("mnm::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenMNMTypeDefault::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation,
                                                       ceil_mode);
  }
  auto outputs =
      LazyTensor::max_pool_nd(bridge::mnm_backend::GetLtcTensor(self), /*spatial_dim_count=*/3,
                              Helpers::I64List(kernel_size), Helpers::I64List(stride),
                              Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor AtenMNMType::max_unpool2d(const at::Tensor& self, const at::Tensor& indices,
                                     at::IntArrayRef output_size) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::mnm_backend::GetLtcTensor(self), bridge::mnm_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size)));
}

at::Tensor AtenMNMType::max_unpool2d_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                              const at::Tensor& indices,
                                              at::IntArrayRef output_size) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size)));
}

at::Tensor AtenMNMType::max_unpool3d(const at::Tensor& self, const at::Tensor& indices,
                                     at::IntArrayRef output_size, at::IntArrayRef stride,
                                     at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::mnm_backend::GetLtcTensor(self), bridge::mnm_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size)));
}

at::Tensor AtenMNMType::max_unpool3d_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                              const at::Tensor& indices,
                                              at::IntArrayRef output_size, at::IntArrayRef stride,
                                              at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(indices),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size)));
}

at::Tensor AtenMNMType::mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::mean(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenMNMType::mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::mean(bridge::mnm_backend::GetLtcTensor(self),
                       lazy_tensors::util::ToVector<lazy_tensors::int64>(dim),
                       /*keep_reduced_dimensions=*/keepdim, dtype));
}

at::Tensor AtenMNMType::min(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::min(bridge::mnm_backend::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::min(const at::Tensor& self, int64_t dim,
                                                    bool keepdim) {
  return AtenMNMTypeDefault::min(self, dim, keepdim);
}

at::Tensor AtenMNMType::minimum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::min(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMNMType::min_out(const at::Tensor& self, int64_t dim,
                                                          bool keepdim, at::Tensor& min,
                                                          at::Tensor& min_indices) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor min_tensor = bridge::mnm_backend::GetLtcTensor(min);
  LazyTensor min_indices_tensor = bridge::mnm_backend::GetLtcTensor(min_indices);
  LazyTensor::min_out(min_tensor, min_indices_tensor, bridge::mnm_backend::GetLtcTensor(self), dim,
                      keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor AtenMNMType::mm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(mat2)) {
    return AtenMNMTypeDefault::mm(self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::mm(/*input=*/bridge::mnm_backend::GetLtcTensor(self),
                     /*weight=*/bridge::mnm_backend::GetLtcTensor(mat2)));
}

at::Tensor AtenMNMType::mse_loss(const at::Tensor& self, const at::Tensor& target,
                                 int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss(bridge::mnm_backend::GetLtcTensor(self),
                                                        bridge::mnm_backend::GetLtcTensor(target),
                                                        reduction));
}

at::Tensor AtenMNMType::mse_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                          const at::Tensor& target, int64_t reduction) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(target), reduction));
}

at::Tensor AtenMNMType::mul(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::mul(xself, xother, dtype);
                    });
}

at::Tensor AtenMNMType::mul(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::mul(xself, other, dtype);
                    });
}

at::Tensor& AtenMNMType::mul_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::mul_(self_tensor, bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()));
    return self;
  }
  return AtenMNMTypeDefault::mul_(self, other);
}

at::Tensor& AtenMNMType::mul_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::mul_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::mul_(self, other);
}

at::Tensor AtenMNMType::mv(const at::Tensor& self, const at::Tensor& vec) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(vec)) {
    return AtenMNMTypeDefault::mv(self, vec);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::mv(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(vec)));
}

at::Tensor& AtenMNMType::mv_out(const at::Tensor& self, const at::Tensor& vec, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) || !at::native::is_floating_point(vec)) {
    return AtenMNMTypeDefault::mv_out(self, vec, out);
  }
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor::mv_out(out_tensor, bridge::mnm_backend::GetLtcTensor(self),
                     bridge::mnm_backend::GetLtcTensor(vec));
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenMNMType::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor input_tensor = bridge::mnm_backend::GetLtcTensor(input);
  const Device& device = input_tensor.GetDevice();
  LazyTensor running_mean_tensor = bridge::GetOrCreateLtcTensor(running_mean, device);
  LazyTensor running_var_tensor = bridge::GetOrCreateLtcTensor(running_var, device);
  auto outputs = LazyTensor::native_batch_norm(
      bridge::mnm_backend::GetLtcTensor(input), bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(bias, device), running_mean_tensor, running_var_tensor, training,
      momentum, eps);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenMNMType::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean, const c10::optional<at::Tensor>& save_invstd,
    bool train, double eps, std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor grad_out_tensor = bridge::mnm_backend::GetLtcTensor(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = LazyTensor::native_batch_norm_backward(
      bridge::mnm_backend::GetLtcTensor(grad_out), bridge::mnm_backend::GetLtcTensor(input),
      bridge::GetOrCreateLtcTensor(weight, device), bridge::GetOrCreateLtcTensor(save_mean, device),
      bridge::GetOrCreateLtcTensor(save_invstd, device), train, eps);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients)) : undefined,
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients)) : undefined,
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients)) : undefined);
}

at::Tensor AtenMNMType::ne(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ne(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor AtenMNMType::ne(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::ne(bridge::mnm_backend::GetLtcTensor(self),
                                                  bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor& AtenMNMType::ne_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::ne_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::ne_(self, other);
}

at::Tensor& AtenMNMType::ne_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::ne_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::ne_(self, other);
}

at::Tensor AtenMNMType::neg(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LTC_CHECK(self.scalar_type() != at::kBool)
      << "Negation, the `-` operator, on a bool tensor is not supported. If "
         "you are trying to invert a mask, use the `~` or `logical_not()` "
         "operator instead.";
  return bridge::AtenFromLtcTensor(LazyTensor::neg(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::neg_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::neg_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::neg_(self);
}

at::Tensor AtenMNMType::nll_loss2d_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                            const at::Tensor& target,
                                            const c10::optional<at::Tensor>& weight,
                                            int64_t reduction, int64_t ignore_index,
                                            const at::Tensor& total_weight) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor = bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::nll_loss2d_backward(bridge::mnm_backend::GetLtcTensor(grad_output), self_tensor,
                                      bridge::mnm_backend::GetLtcTensor(target), weight_tensor,
                                      reduction, ignore_index, total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor total_weight = LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss2d(
          self_tensor, bridge::mnm_backend::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()), reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor AtenMNMType::nll_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                          const at::Tensor& target,
                                          const c10::optional<at::Tensor>& weight,
                                          int64_t reduction, int64_t ignore_index,
                                          const at::Tensor& total_weight) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor weight_tensor = bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor = bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::nll_loss_backward(bridge::mnm_backend::GetLtcTensor(grad_output), self_tensor,
                                    bridge::mnm_backend::GetLtcTensor(target), weight_tensor,
                                    reduction, ignore_index, total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor total_weight = LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss(
          self_tensor, bridge::mnm_backend::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()), reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor AtenMNMType::nonzero(const at::Tensor& self) {
  return AtenMNMTypeDefault::nonzero(self);
}

at::Tensor AtenMNMType::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                             at::ScalarType dtype) {
  return AtenMNMTypeDefault::norm(self, p, dtype);
}

at::Tensor AtenMNMType::norm(const at::Tensor& self, const at::Scalar& p) {
  return AtenMNMTypeDefault::norm(self, p);
}

at::Tensor AtenMNMType::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  return AtenMNMTypeDefault::norm(self, p, dim, keepdim, dtype);
}

at::Tensor AtenMNMType::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim) {
  return AtenMNMTypeDefault::norm(self, p, dim, keepdim);
}

at::Tensor AtenMNMType::normal(const at::Tensor& mean, double std,
                               c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(bridge::mnm_backend::GetLtcTensor(mean), std));
}

at::Tensor AtenMNMType::normal(double mean, const at::Tensor& std,
                               c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(mean, bridge::mnm_backend::GetLtcTensor(std)));
}

at::Tensor AtenMNMType::normal(const at::Tensor& mean, const at::Tensor& std,
                               c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::normal(bridge::mnm_backend::GetLtcTensor(mean),
                                                      bridge::mnm_backend::GetLtcTensor(std)));
}

at::Tensor& AtenMNMType::normal_(at::Tensor& self, double mean, double std,
                                 c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::normal_(self, mean, std, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor AtenMNMType::permute(const at::Tensor& self, at::IntArrayRef dims) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
}

at::Tensor AtenMNMType::pow(const at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_COUNTER("mnm::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenMNMTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(bridge::mnm_backend::GetLtcTensor(self), exponent));
}

at::Tensor AtenMNMType::pow(const at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("mnm::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenMNMTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::pow(bridge::mnm_backend::GetLtcTensor(self),
                                                   bridge::mnm_backend::GetLtcTensor(exponent)));
}

at::Tensor AtenMNMType::pow(const at::Scalar& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("mnm::");
  // xla::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return AtenMNMTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(self, bridge::mnm_backend::GetLtcTensor(exponent)));
}

at::Tensor& AtenMNMType::pow_(at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_COUNTER("mnm::");
  // xla::Pow() doesn't support integer types.
  if (!bridge::IsInteropView(self) && at::native::is_floating_point(self)) {
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::pow_(self_tensor, exponent);
    return self;
  }
  return AtenMNMTypeDefault::pow_(self, exponent);
}

at::Tensor& AtenMNMType::pow_(at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("mnm::");
  // xla::Pow() doesn't support integer types.
  if (!bridge::IsInteropView(self) && at::native::is_floating_point(self)) {
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::pow_(self_tensor, bridge::mnm_backend::GetLtcTensor(exponent));
    return self;
  }
  return AtenMNMTypeDefault::pow_(self, exponent);
}

at::Tensor AtenMNMType::prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::prod(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor AtenMNMType::prod(const at::Tensor& self, int64_t dim, bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::prod(bridge::mnm_backend::GetLtcTensor(self), {dim}, keepdim,
                       PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor& AtenMNMType::put_(at::Tensor& self, const at::Tensor& index, const at::Tensor& source,
                              bool accumulate) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::put_(self_tensor, bridge::mnm_backend::GetLtcTensor(index),
                   bridge::mnm_backend::GetLtcTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::qr(const at::Tensor& self, bool some) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::qr(bridge::mnm_backend::GetLtcTensor(self), some);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

// The value generated should be within (from, to].
at::Tensor& AtenMNMType::random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                                 c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::random_(self, from, to, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
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
at::Tensor& AtenMNMType::random_(at::Tensor& self, int64_t to,
                                 c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::random_(self, to, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_CHECK_GT(to, 0);
  CheckRangeValues(self_tensor.dtype(), 0, to - 1);
  LazyTensor::random_(self_tensor, 0, to);
  return self;
}

// The value generated should be in (self_type_min, self_type_max).
at::Tensor& AtenMNMType::random_(at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::random_(self, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  at::ScalarType dtype = self_tensor.dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  LazyTensor::random_(self_tensor, 0, GetIntegerUpperLimitForType(dtype) + inc);
  return self;
}

at::Tensor AtenMNMType::reciprocal(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::reciprocal(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::reciprocal_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::reciprocal_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::reciprocal_(self);
}

at::Tensor AtenMNMType::reflection_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::reflection_pad2d(bridge::mnm_backend::GetLtcTensor(self),
                                   lazy_tensors::util::ToVector<lazy_tensors::int64>(padding)));
}

at::Tensor AtenMNMType::reflection_pad2d_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::reflection_pad2d_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(padding)));
}

at::Tensor AtenMNMType::relu(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::relu(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::relu_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::relu_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::relu_(self);
}

at::Tensor AtenMNMType::remainder(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::remainder(bridge::mnm_backend::GetLtcTensor(self),
                                                         bridge::mnm_backend::GetLtcTensor(other)));
}

at::Tensor AtenMNMType::remainder(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::remainder(bridge::mnm_backend::GetLtcTensor(self), other));
}

at::Tensor& AtenMNMType::remainder_(at::Tensor& self, const at::Tensor& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::remainder_(self_tensor, bridge::mnm_backend::GetLtcTensor(other));
    return self;
  }
  return AtenMNMTypeDefault::remainder_(self, other);
}

at::Tensor& AtenMNMType::remainder_(at::Tensor& self, const at::Scalar& other) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::remainder_(self_tensor, other);
    return self;
  }
  return AtenMNMTypeDefault::remainder_(self, other);
}

at::Tensor AtenMNMType::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  return AtenMNMTypeDefault::repeat(self, repeats);
}

at::Tensor AtenMNMType::replication_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d(
      bridge::mnm_backend::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor AtenMNMType::replication_pad1d_backward(const at::Tensor& grad_output,
                                                   const at::Tensor& self,
                                                   at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

at::Tensor AtenMNMType::replication_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d(
      bridge::mnm_backend::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor AtenMNMType::replication_pad2d_backward(const at::Tensor& grad_output,
                                                   const at::Tensor& self,
                                                   at::IntArrayRef padding) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

const at::Tensor& AtenMNMType::resize_(const at::Tensor& self, at::IntArrayRef size,
                                       c10::optional<at::MemoryFormat> memory_format) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::resize_(self_tensor, Helpers::I64List(size));
    return self;
  }
  return AtenMNMTypeDefault::resize_(self, size, memory_format);
}

at::Tensor AtenMNMType::round(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::round(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::round_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::round_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::round_(self);
}

at::Tensor AtenMNMType::rrelu_with_noise(const at::Tensor& self, const at::Tensor& noise,
                                         const at::Scalar& lower, const at::Scalar& upper,
                                         bool training, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    LTC_CHECK_EQ(training, false);
    return AtenMNMTypeDefault::rrelu_with_noise(self, noise, lower, upper, training, generator);
  }
  LazyTensor noise_tensor = bridge::mnm_backend::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise(
      bridge::mnm_backend::GetLtcTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor AtenMNMType::rrelu_with_noise_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self, const at::Tensor& noise,
                                                  const at::Scalar& lower, const at::Scalar& upper,
                                                  bool training, bool self_is_result) {
  LTC_FN_COUNTER("mnm::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  LTC_CHECK(!self_is_result || negative_slope > 0.0);
  LazyTensor noise_tensor = bridge::mnm_backend::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor AtenMNMType::rsqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::rsqrt(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::rsqrt_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::rsqrt_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::rsqrt_(self);
}

at::Tensor AtenMNMType::rsub(const at::Tensor& self, const at::Tensor& other,
                             const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::rsub(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenMNMType::rsub(const at::Tensor& self, const at::Scalar& other,
                             const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return bridge::AtenFromLtcTensor(
      LazyTensor::rsub(bridge::mnm_backend::GetLtcTensor(self), other, alpha));
}

at::Tensor& AtenMNMType::scatter_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                  const at::Tensor& src) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::scatter_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index),
                       bridge::mnm_backend::GetLtcTensor(src));
  return self;
}

at::Tensor& AtenMNMType::scatter_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                  const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::scatter_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index), value);
  return self;
}

at::Tensor& AtenMNMType::scatter_add_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                      const at::Tensor& src) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::scatter_add_(self_tensor, dim, bridge::mnm_backend::GetLtcTensor(index),
                           bridge::mnm_backend::GetLtcTensor(src));
  return self;
}

at::Tensor AtenMNMType::select(const at::Tensor& self, int64_t dim, int64_t index) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::select(bridge::mnm_backend::GetLtcTensor(self), dim, index));
}

at::Tensor& AtenMNMType::silu_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor out_tensor = bridge::mnm_backend::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::silu_out(self_tensor, out_tensor);
  return out;
}

at::Tensor AtenMNMType::sigmoid(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sigmoid(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::sigmoid_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sigmoid_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::sigmoid_(self);
}

at::Tensor AtenMNMType::sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sigmoid_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(output)));
}

at::Tensor AtenMNMType::sign(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sign(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::sign_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sign_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::sign_(self);
}

at::Tensor AtenMNMType::sin(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sin(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::sin_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sin_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::sin_(self);
}

at::Tensor AtenMNMType::sinh(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sinh(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::sinh_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sinh_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::sinh_(self);
}

at::Tensor AtenMNMType::slice(const at::Tensor& self, int64_t dim, c10::optional<int64_t> start,
                              c10::optional<int64_t> end, int64_t step) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  return bridge::AtenFromLtcTensor(
      LazyTensor::slice(bridge::mnm_backend::GetLtcTensor(self), dim, start_val, end_val, step));
}

at::Tensor AtenMNMType::smooth_l1_loss(const at::Tensor& self, const at::Tensor& target,
                                       int64_t reduction, double beta) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::smooth_l1_loss(bridge::mnm_backend::GetLtcTensor(self),
                                 bridge::mnm_backend::GetLtcTensor(target), reduction, beta));
}

at::Tensor AtenMNMType::smooth_l1_loss_backward(const at::Tensor& grad_output,
                                                const at::Tensor& self, const at::Tensor& target,
                                                int64_t reduction, double beta) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::smooth_l1_loss_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      bridge::mnm_backend::GetLtcTensor(target), reduction, beta));
}

at::Tensor AtenMNMType::softplus(const at::Tensor& self, const at::Scalar& beta,
                                 const at::Scalar& threshold) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softplus(bridge::mnm_backend::GetLtcTensor(self), beta, threshold));
}

at::Tensor AtenMNMType::softplus_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                          const at::Scalar& beta, const at::Scalar& threshold,
                                          const at::Tensor& output) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::softplus_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self), beta,
      threshold, bridge::mnm_backend::GetLtcTensor(output)));
}

at::Tensor AtenMNMType::softshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink(bridge::mnm_backend::GetLtcTensor(self), lambda));
}

at::Tensor AtenMNMType::softshrink_backward(const at::Tensor& grad_out, const at::Tensor& self,
                                            const at::Scalar& lambda) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink_backward(bridge::mnm_backend::GetLtcTensor(grad_out),
                                      bridge::mnm_backend::GetLtcTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::sort(const at::Tensor& self, int64_t dim,
                                                     bool descending) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::topk(bridge::mnm_backend::GetLtcTensor(self), self.size(dim), dim,
                                  descending, true);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

std::vector<at::Tensor> AtenMNMType::split(const at::Tensor& self, int64_t split_size,
                                           int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  auto xla_tensors = LazyTensor::split(bridge::mnm_backend::GetLtcTensor(self), split_size, dim);
  return bridge::AtenFromLtcTensors(xla_tensors);
}

std::vector<at::Tensor> AtenMNMType::split_with_sizes(const at::Tensor& self,
                                                      at::IntArrayRef split_sizes, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  auto xla_tensors = LazyTensor::split_with_sizes(bridge::mnm_backend::GetLtcTensor(self),
                                                  Helpers::I64List(split_sizes), dim);
  return bridge::AtenFromLtcTensors(xla_tensors);
}

at::Tensor AtenMNMType::sqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::sqrt(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::sqrt_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sqrt_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::sqrt_(self);
}

at::Tensor AtenMNMType::squeeze(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::squeeze(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::squeeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::squeeze(bridge::mnm_backend::GetLtcTensor(self), dim));
}

at::Tensor& AtenMNMType::squeeze_(at::Tensor& self) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  TORCH_MNM_VLOG(3) << "XLA squeeze_ :"
                    << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_();
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens, xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor& AtenMNMType::squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  TORCH_MNM_VLOG(3) << "XLA squeeze_ :"
                    << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_(dim);
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens, xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor AtenMNMType::stack(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::stack(bridge::mnm_backend::GetLtcTensors(tensors), dim));
}

at::Tensor AtenMNMType::std(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, /*correction=*/unbiased ? 1 : 0));
}

at::Tensor AtenMNMType::std(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
                            bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::std(bridge::mnm_backend::GetLtcTensor(self),
                      lazy_tensors::util::ToVector<lazy_tensors::int64>(dim), keepdim,
                      /*correction=*/unbiased ? 1 : 0));
}

at::Tensor AtenMNMType::std(const at::Tensor& self, c10::optional<at::IntArrayRef> dim,
                            c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      self_tensor,
      dim ? lazy_tensors::util::ToVector<lazy_tensors::int64>(*dim)
          : lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      keepdim, correction ? *correction : 1));
}

at::Tensor AtenMNMType::sub(const at::Tensor& self, const at::Tensor& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother, at::ScalarType dtype) {
                      return LazyTensor::sub(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenMNMType::sub(const at::Tensor& self, const at::Scalar& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("mnm::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other, at::ScalarType dtype) {
                      return LazyTensor::sub(xself, other, alpha, dtype);
                    });
}

at::Tensor& AtenMNMType::sub_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    at::native::alpha_check(at::result_type(self, other), alpha);
    CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sub_(self_tensor, bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()),
                     alpha);
    return self;
  }
  return AtenMNMTypeDefault::sub_(self, other, alpha);
}

at::Tensor& AtenMNMType::sub_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    CheckBinaryOpTypePromotion(self, self, other);
    CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::sub_(self_tensor, other, alpha);
    return self;
  }
  return AtenMNMTypeDefault::sub_(self, other, alpha);
}

at::Tensor AtenMNMType::sum(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::sum(
      self_tensor, lazy_tensors::util::Iota<lazy_tensors::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenMNMType::sum(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                            c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sum(bridge::mnm_backend::GetLtcTensor(self),
                      lazy_tensors::util::ToVector<lazy_tensors::int64>(dim), keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenMNMType::svd(const at::Tensor& self, bool some,
                                                                bool compute_uv) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::svd(bridge::mnm_backend::GetLtcTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)),
                         bridge::AtenFromLtcTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::symeig(const at::Tensor& self, bool eigenvectors,
                                                       bool upper) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::symeig(bridge::mnm_backend::GetLtcTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor AtenMNMType::t(const at::Tensor& self) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::mnm_backend::GetLtcTensor(self), 0, 1));
}

at::Tensor& AtenMNMType::t_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::transpose_(self_tensor, 0, 1);
    return self;
  }
  return AtenMNMTypeDefault::t_(self);
}

at::Tensor AtenMNMType::take(const at::Tensor& self, const at::Tensor& index) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::take(bridge::mnm_backend::GetLtcTensor(self),
                                                    bridge::mnm_backend::GetLtcTensor(index)));
}

at::Tensor AtenMNMType::tan(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::tan(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::tan_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::tan_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::tan_(self);
}

at::Tensor AtenMNMType::tanh(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::tanh(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::tanh_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::tanh_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::tanh_(self);
}

at::Tensor AtenMNMType::tanh_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::tanh_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(output)));
}

at::Tensor AtenMNMType::threshold(const at::Tensor& self, const at::Scalar& threshold,
                                  const at::Scalar& value) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold(
      bridge::mnm_backend::GetLtcTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor& AtenMNMType::threshold_(at::Tensor& self, const at::Scalar& threshold,
                                    const at::Scalar& value) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::threshold_(self_tensor, threshold.to<double>(), value.to<double>());
    return self;
  }
  return AtenMNMTypeDefault::threshold_(self, threshold, value);
}

at::Tensor AtenMNMType::threshold_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                           const at::Scalar& threshold) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold_backward(
      bridge::mnm_backend::GetLtcTensor(grad_output), bridge::mnm_backend::GetLtcTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::topk(const at::Tensor& self, int64_t k, int64_t dim,
                                                     bool largest, bool sorted) {
  LTC_FN_COUNTER("mnm::");
  auto results = LazyTensor::topk(bridge::mnm_backend::GetLtcTensor(self), k, dim, largest, sorted);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor AtenMNMType::trace(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::trace(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor AtenMNMType::transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::mnm_backend::GetLtcTensor(self), dim0, dim1));
}

at::Tensor& AtenMNMType::transpose_(at::Tensor& self, int64_t dim0, int64_t dim1) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenMNMType::triangular_solve(const at::Tensor& b,
                                                                 const at::Tensor& A, bool upper,
                                                                 bool transpose,
                                                                 bool unitriangular) {
  LTC_FN_COUNTER("mnm::");
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = LazyTensor::triangular_solve(bridge::mnm_backend::GetLtcTensor(b),
                                              bridge::mnm_backend::GetLtcTensor(A),
                                              /*left_side=*/true, upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor AtenMNMType::tril(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::tril(bridge::mnm_backend::GetLtcTensor(self), diagonal));
}

at::Tensor& AtenMNMType::tril_(at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::tril_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenMNMType::triu(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::triu(bridge::mnm_backend::GetLtcTensor(self), diagonal));
}

at::Tensor& AtenMNMType::triu_(at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::triu_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenMNMType::trunc(const at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::trunc(bridge::mnm_backend::GetLtcTensor(self)));
}

at::Tensor& AtenMNMType::trunc_(at::Tensor& self) {
  if (!bridge::IsInteropView(self)) {
    LTC_FN_COUNTER("mnm::");
    LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
    LazyTensor::trunc_(self_tensor);
    return self;
  }
  return AtenMNMTypeDefault::trunc_(self);
}

std::vector<at::Tensor> AtenMNMType::unbind(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensors(
      LazyTensor::unbind(bridge::mnm_backend::GetLtcTensor(self), dim));
}

at::Tensor& AtenMNMType::uniform_(at::Tensor& self, double from, double to,
                                  c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("mnm::");
  if (generator.has_value() && generator->defined()) {
    return AtenMNMTypeDefault::uniform_(self, from, to, generator);
  }
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor AtenMNMType::unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::unsqueeze(bridge::mnm_backend::GetLtcTensor(self), dim));
}

at::Tensor& AtenMNMType::unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenMNMType::upsample_bilinear2d(const at::Tensor& self, at::IntArrayRef output_size,
                                            bool align_corners, c10::optional<double> scales_h,
                                            c10::optional<double> scales_w) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return AtenMNMTypeDefault::upsample_bilinear2d(self, output_size, align_corners, scales_h,
                                                   scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d(
      self_tensor, lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size), align_corners));
}

at::Tensor AtenMNMType::upsample_bilinear2d_backward(const at::Tensor& grad_output,
                                                     at::IntArrayRef output_size,
                                                     at::IntArrayRef input_size, bool align_corners,
                                                     c10::optional<double> scales_h,
                                                     c10::optional<double> scales_w) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor grad_output_tensor = bridge::mnm_backend::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return AtenMNMTypeDefault::upsample_bilinear2d_backward(grad_output, output_size, input_size,
                                                            align_corners, scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d_backward(
      grad_output_tensor, lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(input_size), align_corners));
}

at::Tensor AtenMNMType::upsample_nearest2d(const at::Tensor& input,
                                           c10::optional<at::IntArrayRef> output_size,
                                           c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor input_tensor = bridge::mnm_backend::GetLtcTensor(input);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (input_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenMNMTypeDefault::upsample_nearest2d(input, output_size, scale_factors);
  }
  absl::Span<const lazy_tensors::int64> input_dims = input_tensor.shape().get().dimensions();
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      input_tensor, GetOutputSizeWithScale(input_dims, scale_factors, output_size)));
}

at::Tensor AtenMNMType::upsample_nearest2d_backward(
    const at::Tensor& grad_output, c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor grad_output_tensor = bridge::mnm_backend::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenMNMTypeDefault::upsample_nearest2d_backward(grad_output, output_size, input_size,
                                                           scale_factors);
  }
  std::vector<lazy_tensors::int64> input_dim =
      lazy_tensors::util::ToVector<lazy_tensors::int64>(input_size);
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor, GetOutputSizeWithScale(input_dim, scale_factors, output_size),
      input_dim));
}

at::Tensor AtenMNMType::upsample_nearest2d(const at::Tensor& self, at::IntArrayRef output_size,
                                           c10::optional<double> scales_h,
                                           c10::optional<double> scales_w) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return AtenMNMTypeDefault::upsample_nearest2d(self, output_size, scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      self_tensor, lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size)));
}

at::Tensor AtenMNMType::upsample_nearest2d_backward(const at::Tensor& grad_output,
                                                    at::IntArrayRef output_size,
                                                    at::IntArrayRef input_size,
                                                    c10::optional<double> scales_h,
                                                    c10::optional<double> scales_w) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor grad_output_tensor = bridge::mnm_backend::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return AtenMNMTypeDefault::upsample_nearest2d_backward(grad_output, output_size, input_size,
                                                           scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor, lazy_tensors::util::ToVector<lazy_tensors::int64>(output_size),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(input_size)));
}

at::Tensor AtenMNMType::var(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(bridge::mnm_backend::GetLtcTensor(self),
                      lazy_tensors::util::Iota<lazy_tensors::int64>(
                          bridge::mnm_backend::GetLtcTensor(self).shape().get().rank()),
                      /*correction=*/unbiased ? 1 : 0,
                      /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenMNMType::var(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
                            bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::var(self_tensor, Helpers::I64List(dim),
                                                   /*correction=*/unbiased ? 1 : 0, keepdim));
}

at::Tensor AtenMNMType::var(const at::Tensor& self, c10::optional<at::IntArrayRef> dim,
                            c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(self_tensor,
                      dim ? Helpers::I64List(*dim)
                          : lazy_tensors::util::Iota<lazy_tensors::int64>(
                                bridge::mnm_backend::GetLtcTensor(self).shape().get().rank()),
                      correction ? *correction : 1, keepdim));
}

at::Tensor AtenMNMType::view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LTC_FN_COUNTER("mnm::");
  return bridge::AtenFromLtcTensor(LazyTensor::view(self_tensor, Helpers::I64List(size)));
}

at::Tensor& AtenMNMType::zero_(at::Tensor& self) {
  LTC_FN_COUNTER("mnm::");
  LazyTensor self_tensor = bridge::mnm_backend::GetLtcTensor(self);
  LazyTensor::zero_(self_tensor);
  return self;
}

at::Scalar AtenMNMType::_local_scalar_dense(const at::Tensor& self) {
  return AtenMNMTypeDefault::_local_scalar_dense(self);
}

void AtenMNMType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

}  // namespace torch_lazy_tensors
