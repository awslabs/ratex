/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main LazyTensor
// class.

namespace torch_lazy_tensors {
namespace tensor_ops {

LazyTensor Cross(const LazyTensor& input, const LazyTensor& other, c10::optional<int64_t> dim);

LazyTensor KlDivBackward(const LazyTensor& grad_output, const LazyTensor& input,
                         const LazyTensor& target, ReductionMode reduction, bool log_target);

LazyTensor MakeMatrixWithDiagonal(const LazyTensor& input, int64_t diagonal);

LazyTensor SmoothL1Loss(const LazyTensor& input, const LazyTensor& target, ReductionMode reduction,
                        double beta);

LazyTensor SmoothL1LossBackward(const LazyTensor& grad_output, const LazyTensor& input,
                                const LazyTensor& target, ReductionMode reduction, double beta);

LazyTensor Softplus(const LazyTensor& input, const at::Scalar& beta, const at::Scalar& threshold);

LazyTensor SoftplusBackward(const LazyTensor& grad_output, const LazyTensor& input,
                            const at::Scalar& beta, const at::Scalar& threshold);

LazyTensor Select(const LazyTensor& input, int64_t dim, int64_t index);

LazyTensor EmbeddingDenseBackward(const LazyTensor& grad_output, const LazyTensor& indices,
                                  int64_t num_weights, int64_t padding_idx,
                                  bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
