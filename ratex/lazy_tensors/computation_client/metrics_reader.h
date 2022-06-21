/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTATION_CLIENT_METRICS_READER_H_
#define COMPUTATION_CLIENT_METRICS_READER_H_

#include <string>

namespace lazy_tensors {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

}  // namespace metrics_reader
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_METRICS_READER_H_
