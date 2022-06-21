/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unordered_map>

#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "ratex/csrc/aten_cpu_fallback.h"
#include "ratex/csrc/utils/ratex_logging.h"

namespace torch_lazy_tensors {

void raf_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  LTC_COUNTER("aten::" + c10::toString(op.operator_name()), 1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      RATEX_VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, Lazy, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&raf_cpu_fallback>());
}

// Other fallback functions.
at::Tensor& AtenRAFTypeDefault::add_(at::Tensor& self, const at::Scalar& other,
                                     const at::Scalar& alpha) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::add_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].add_(other, alpha);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                           c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::bernoulli_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, p};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].bernoulli_(ltc_atens[1], generator);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::bernoulli_(at::Tensor& self, double p,
                                           c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::bernoulli_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].bernoulli_(p, generator);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::div_(at::Tensor& self, const at::Tensor& other,
                                     c10::optional<c10::string_view> rounding_mode) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::div_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].div_(ltc_atens[1], rounding_mode);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::div_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::div_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].div_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::eq_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::eq_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].eq_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::eq_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::eq_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].eq_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::fill_(at::Tensor& self, const at::Scalar& value) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::fill_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  at::fill_(ltc_atens[0], value);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::fill_(at::Tensor& self, const at::Tensor& value) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::fill_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, value};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  at::fill_(ltc_atens[0], ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::fmod_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::fmod_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].fmod_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::fmod_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::fmod_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].fmod_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::ge_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::ge_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].ge_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::ge_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::ge_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].ge_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::gt_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::gt_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].gt_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::gt_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::gt_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].gt_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::le_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::le_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].le_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::le_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::le_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].le_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::lt_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::lt_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].lt_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::lt_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::lt_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].lt_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenRAFTypeDefault::min(const at::Tensor& self, int64_t dim,
                                                           bool keepdim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::min", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::min(ltc_atens[0], dim, keepdim);
  return std::tuple<at::Tensor, at::Tensor>(
      bridge::CreateLtcTensor(std::get<0>(x_result), bridge::GetLtcDevice(self)),
      bridge::CreateLtcTensor(std::get<1>(x_result), bridge::GetLtcDevice(self)));
}

at::Tensor& AtenRAFTypeDefault::mul_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::mul_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].mul_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::mul_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::mul_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].mul_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::ne_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::ne_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].ne_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::ne_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::ne_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].ne_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor AtenRAFTypeDefault::normal(double mean, const at::Tensor& std,
                                      c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::normal", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {std};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::normal(mean, ltc_atens[0], generator);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(std));
}

at::Tensor AtenRAFTypeDefault::normal(const at::Tensor& mean, const at::Tensor& std,
                                      c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::normal", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {mean, std};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::normal(ltc_atens[0], ltc_atens[1], generator);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(mean));
}

at::Tensor AtenRAFTypeDefault::normal(const at::Tensor& mean, double std,
                                      c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::normal", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {mean};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::normal(ltc_atens[0], std, generator);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(mean));
}

at::Tensor AtenRAFTypeDefault::pow(const at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::pow", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, exponent};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::pow(ltc_atens[0], ltc_atens[1]);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(self));
}

at::Tensor AtenRAFTypeDefault::pow(const at::Scalar& self, const at::Tensor& exponent) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::pow", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {exponent};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::pow(self, ltc_atens[0]);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(exponent));
}

at::Tensor AtenRAFTypeDefault::pow(const at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::pow", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::pow(ltc_atens[0], exponent);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(self));
}

at::Tensor& AtenRAFTypeDefault::pow_(at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::pow_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, exponent};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].pow_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::pow_(at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::pow_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].pow_(exponent);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                                        c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::random_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].random_(from, to, generator);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::random_(at::Tensor& self, int64_t to,
                                        c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::random_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].random_(to, generator);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::remainder_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::remainder_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].remainder_(other);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::remainder_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::remainder_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].remainder_(ltc_atens[1]);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::sub_(at::Tensor& self, const at::Tensor& other,
                                     const at::Scalar& alpha) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::sub_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self, other};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].sub_(ltc_atens[1], alpha);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor& AtenRAFTypeDefault::sub_(at::Tensor& self, const at::Scalar& other,
                                     const at::Scalar& alpha) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::sub_", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {self};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  ltc_atens[0].sub_(other, alpha);
  std::vector<size_t> ltc_atens_update_indices = {0};
  bridge::LtcUpdateTensors(ltc_atens_tensors, ltc_atens, ltc_atens_update_indices);
  return self;
}

at::Tensor AtenRAFTypeDefault::upsample_nearest2d(
    const at::Tensor& input, at::OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::upsample_nearest2d", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {input};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result = at::upsample_nearest2d(ltc_atens[0], output_size.value(), scale_factors);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(input));
}

at::Tensor AtenRAFTypeDefault::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::OptionalIntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::upsample_nearest2d_backward", 1);
  std::vector<at::Tensor> ltc_atens_tensors = {grad_output};
  auto ltc_atens = bridge::LtcCreateTensorList(ltc_atens_tensors);
  auto&& x_result =
      at::upsample_nearest2d_backward(ltc_atens[0], output_size.value(), input_size, scale_factors);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(grad_output));
}

}  // namespace torch_lazy_tensors
