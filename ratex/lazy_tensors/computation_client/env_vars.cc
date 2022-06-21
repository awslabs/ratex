/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensors/computation_client/env_vars.h"

namespace lazy_tensors {
namespace env {

const char* const kEnvNumTpu = "TPU_NUM_DEVICES";
const char* const kEnvNumGpu = "GPU_NUM_DEVICES";
const char* const kEnvNumCpu = "CPU_NUM_DEVICES";
const char* const kEnvLocalWorker = "XRT_LOCAL_WORKER";
const char* const kEnvTpuConfig = "XRT_TPU_CONFIG";
const char* const kEnvMeshService = "XRT_MESH_SERVICE_ADDRESS";
const char* const kEnvWorldSize = "XRT_SHARD_WORLD_SIZE";
const char* const kEnvMpDevice = "XRT_MULTI_PROCESSING_DEVICE";
const char* const kEnvHostOrdinal = "XRT_HOST_ORDINAL";
const char* const kEnvShardOrdinal = "XRT_SHARD_ORDINAL";
const char* const kEnvStartService = "XRT_START_LOCAL_SERVER";
const char* const kEnvTpuvmMode = "TPUVM_MODE";

}  // namespace env
}  // namespace lazy_tensors
