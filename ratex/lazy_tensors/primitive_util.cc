/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensors/primitive_util.h"

namespace lazy_tensors {
namespace primitive_util {

bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != PrimitiveType::INVALID && primitive_type != PrimitiveType::TUPLE;
}

}  // namespace primitive_util
}  // namespace lazy_tensors
