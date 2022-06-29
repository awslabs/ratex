/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
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
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace compiler {

inline lazy_tensors::Shape InferUnary(const ir::Node* node) {
  const ir::Output& input = node->operand(0);
  return input.shape();
}

inline lazy_tensors::Shape InferGenericSlice(const ir::ops::GenericSlice* node) {
  const ir::Output& input = node->operand(0);
  lazy_tensors::Shape ret = input.shape();
  LTC_CHECK_EQ(ret.dimensions_size(), node->sizes().size());
  for (int i = 0; i < ret.dimensions_size(); ++i) {
    ret.set_dimensions(i, node->sizes()[i]);
  }
  return ret;
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
