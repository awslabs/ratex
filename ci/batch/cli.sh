#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The CLI scripts for running tasks. This is supposed to be used by
# automated CI so it assumes
#   1. The script is run in the repo root folder.
#   2. The repo has been well-configured.
# Example usages:
#   bash cli.sh compile
#   bash cli.sh unit_test GPU
#   bash cli.sh update_docker latest
set -e
set -o pipefail

# Build the docker image and push to docker hub.
function update_docker() {
    TAG=$1

    cd docker
    bash ./build.sh ci_gpu

    # Push the image
    bash ./push.sh ci_gpu $TAG
}

# Compile, build the wheel and install.
function compile() {
    JOB_TAG=$1
    echo "======================="
    echo "[CLI] Compilation start"
    echo "======================="

    # Load ccache if available.
    bash ./ci/batch/backup-ccache.sh download GPU $JOB_TAG || true

    # Compile. Note that compilation errors will not result in crash in this function.
    # We use return exit code to let the caller decide the action.
    BUILD_TYPE=Release USE_CUTLASS=ON bash ./scripts/build_third_party.sh || true
    bash ./scripts/build_razor.sh || true
    RET=$?

    # Backup the ccache.
    bash ./ci/batch/backup-ccache.sh upload GPU $JOB_TAG || true
    echo "======================"
    echo "[CLI] Compilation done"
    echo "======================"
    return $RET
}

# Run linting (compilation is required).
function lint() {
    echo "============="
    echo "[CLI] Linting"
    echo "============="
    bash ./scripts/lint/check-lint.sh
    echo "================"
    echo "[CLI] Formatting"
    echo "================"
    bash ./scripts/lint/check-format.sh
    echo "======================"
    echo "[CLI] Checking license"
    echo "======================"
    python3 ./scripts/lint/check-license-header.py origin/main
    return 0
}

# Run unit tests (compilation is required).
function unit_test() {
    DEVICE=$1
    export ENABLE_PARAM_ALIASING=true
    export RAZOR_CACHE_DIR=""

    if [[ $DEVICE == "GPU" ]]; then
        export RAZOR_DEVICE=GPU
    elif [[ $DEVICE == "CPU" ]]; then
        export RAZOR_DEVICE=CPU
    else
        echo "Unrecognized devic: $DEVICE"
        exit 1
    fi
    echo "=========================================="
    echo "[CLI] Running unit tests with environment:"
    echo "  RAZOR_DEVICE=$RAZOR_DEVICE"
    echo "  ENABLE_PARAM_ALIASING=$ENABLE_PARAM_ALIASING"
    echo "  RAZOR_CACHE_DIR=$RAZOR_CACHE_DIR"
    echo "=========================================="
    time python3 -m pytest tests/python
    echo "=========================================="
    echo "[CLI] Unit tests on $RAZOR_DEVICE are done"
    echo "=========================================="
    return 0
}

# Run compatibility unit tests for PyTorch 1.11
function unit_test_torch_1_11() {
    DEVICE=$1
    export ENABLE_PARAM_ALIASING=true
    export RAZOR_CACHE_DIR=""

    if [[ $DEVICE == "GPU" ]]; then
        export RAZOR_DEVICE=GPU
    elif [[ $DEVICE == "CPU" ]]; then
        export RAZOR_DEVICE=CPU
    else
        echo "Unrecognized devic: $DEVICE"
        exit 1
    fi
    echo "==========================================================="
    echo "[CLI] Running unit tests for PyTorch 1.11 with environment:"
    echo "  RAZOR_DEVICE=$RAZOR_DEVICE"
    echo "  ENABLE_PARAM_ALIASING=$ENABLE_PARAM_ALIASING"
    echo "  RAZOR_CACHE_DIR=$RAZOR_CACHE_DIR"
    echo "==========================================================="
    time python3 -m pytest tests/python/ -m torch_1_11_test
    echo "=========================================="
    echo "[CLI] Unit tests on $RAZOR_DEVICE are done"
    echo "=========================================="
    return 0
}

# Update CI badge
function update_ci_badge() {
    PR=$1
    if [ ! -z "$PR" ]; then
        echo "PR number is provided, meaning this is a PR build. Skip updating CI badge."
        exit 0
    fi
    RAZOR_VERSION=$(git rev-parse --short HEAD)
    echo "Razor version: ${RAZOR_VERSION}"
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch version: ${TORCH_VERSION}"
    echo "$RAZOR_VERSION (PyTorch $TORCH_VERSION)" > razor-ci-badge-last-pass.txt
    aws s3 cp razor-ci-badge-last-pass.txt s3://ci-razor/razor-ci-badge-last-pass.txt
}

# Run the function from command line.
if declare -f "$1" > /dev/null
then
    # Call arguments verbatim if the function exists
    "$@"
else
    # Show a helpful error
    echo "'$1' is not a known function name" >&2
    exit 1
fi
