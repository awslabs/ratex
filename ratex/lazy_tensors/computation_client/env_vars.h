/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTATION_CLIENT_ENV_VARS_H_
#define COMPUTATION_CLIENT_ENV_VARS_H_

namespace lazy_tensors {
namespace env {

extern const char* const kEnvNumTpu;
extern const char* const kEnvNumGpu;
extern const char* const kEnvNumCpu;
extern const char* const kEnvLocalWorker;
extern const char* const kEnvTpuConfig;
extern const char* const kEnvMeshService;
extern const char* const kEnvWorldSize;
extern const char* const kEnvMpDevice;
extern const char* const kEnvHostOrdinal;
extern const char* const kEnvShardOrdinal;
extern const char* const kEnvStartService;
extern const char* const kEnvTpuvmMode;

}  // namespace env
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_ENV_VARS_H_
