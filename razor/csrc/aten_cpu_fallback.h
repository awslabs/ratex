/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>
#include <c10/util/OptionalArrayRef.h>

namespace torch_lazy_tensors {

void raf_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

#define FALLBACK_ATEN_OP(op_name, args...) \
  at::native::call_fallback_fn<&raf_cpu_fallback, ATEN_OP(op_name)>::call(args)

class AtenRAFTypeDefault {
 public:
  static at::Tensor& add_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
  static at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p,
                                c10::optional<at::Generator> generator);
  static at::Tensor& bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> generator);
  static at::Tensor& div_(at::Tensor& self, const at::Tensor& other,
                          c10::optional<c10::string_view> rounding_mode);
  static at::Tensor& div_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& eq_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& eq_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& fill_(at::Tensor& self, const at::Scalar& value);
  static at::Tensor& fill_(at::Tensor& self, const at::Tensor& value);
  static at::Tensor& fmod_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& ge_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& ge_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& gt_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& gt_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& le_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& le_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& lt_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& lt_(at::Tensor& self, const at::Scalar& other);
  static std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, int64_t dim, bool keepdim);
  static at::Tensor& mul_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& mul_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& ne_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& ne_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor normal(double mean, const at::Tensor& std,
                           c10::optional<at::Generator> generator);
  static at::Tensor normal(const at::Tensor& mean, const at::Tensor& std,
                           c10::optional<at::Generator> generator);
  static at::Tensor normal(const at::Tensor& mean, double std,
                           c10::optional<at::Generator> generator);
  static at::Tensor pow(const at::Tensor& self, const at::Tensor& exponent);
  static at::Tensor pow(const at::Scalar& self, const at::Tensor& exponent);
  static at::Tensor pow(const at::Tensor& self, const at::Scalar& exponent);
  static at::Tensor& pow_(at::Tensor& self, const at::Tensor& exponent);
  static at::Tensor& pow_(at::Tensor& self, const at::Scalar& exponent);
  static at::Tensor& random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                             c10::optional<at::Generator> generator);
  static at::Tensor& random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> generator);
  static at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other);
  static at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other);
  static at::Tensor& sub_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
  static at::Tensor& sub_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
  static at::Tensor upsample_nearest2d(const at::Tensor& input,
                                       at::OptionalIntArrayRef output_size,
                                       c10::optional<at::ArrayRef<double>> scale_factors);
  static at::Tensor upsample_nearest2d_backward(const at::Tensor& grad_output,
                                                at::OptionalIntArrayRef output_size,
                                                at::IntArrayRef input_size,
                                                c10::optional<at::ArrayRef<double>> scale_factors);
};

}  // namespace torch_lazy_tensors
