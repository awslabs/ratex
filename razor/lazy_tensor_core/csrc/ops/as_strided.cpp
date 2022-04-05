/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/as_strided.h"

#include <algorithm>

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AsStrided::AsStrided(const Value& input, std::vector<int64_t> size, std::vector<int64_t> stride,
                     int64_t storage_offset)
    : Node(
          ir::OpKind(at::aten::as_strided), {input},
          [&]() { return lazy_tensors::ShapeUtil::MakeShape(input.shape().element_type(), size); },
          /*num_outputs=*/1, lazy_tensors::util::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {
}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << "), stride=("
     << absl::StrJoin(stride_, ", ") << "), storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStrided::Clone(OpList operands) const {
  return MakeNode<AsStrided>(operands.at(0), size_, stride_, storage_offset_);
}

bool AsStrided::StrideIsSupported(const lazy_tensors::Shape& input_shape,
                                  lazy_tensors::Span<const int64_t> size,
                                  lazy_tensors::Span<const int64_t> stride,
                                  int64_t storage_offset) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> AsStrided::GetArrayStridePermutation(lazy_tensors::Span<const int64_t> stride,
                                                          lazy_tensors::Span<const int64_t> size) {
  std::vector<int64_t> permutation = lazy_tensors::util::Iota<int64_t>(stride.size());
  std::sort(permutation.begin(), permutation.end(),
            [&](int64_t a, int64_t b) { return stride[a] > stride[b]; });
  return permutation;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
