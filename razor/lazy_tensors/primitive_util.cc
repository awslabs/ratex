/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#include "lazy_tensors/primitive_util.h"

namespace lazy_tensors {
namespace primitive_util {

bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != PrimitiveType::INVALID && primitive_type != PrimitiveType::TUPLE;
}

}  // namespace primitive_util
}  // namespace lazy_tensors
