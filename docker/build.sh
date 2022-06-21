#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the docker image
#
# Usage: build.sh <CONTAINER_TYPE> [--dockerfile <DOCKERFILE_PATH>] [--build-arg <NAME>=<VALUE>]
#
# CONTAINER_TYPE: Type of the docker container used the run the build
#                 (e.g., ci_gpu)
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.  If
#                  this optional value is not supplied (via the --dockerfile
#                  flag), will use Dockerfile.CONTAINER_TYPE in default
# DOCKERBUILD_ARG: (Optional) Additional docker build args.
DOCKER_BINARY="docker"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

if [[ "$1" == "--dockerfile" ]]; then
    DOCKERFILE_PATH="$2"
    DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
    echo "Using custom Dockerfile path: ${DOCKERFILE_PATH}"
    echo "Using custom docker build context path: ${DOCKER_CONTEXT_PATH}"
    shift 2
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

# Docker build args
DOCKER_BUILD_ARGS=""
if [[ "$1" == "--build-arg" ]]; then
    DOCKER_BUILD_ARGS="$1 $2"
    echo "Build args: ${DOCKER_BUILD_ARGS}"
    shift 2
fi

# Validate command line arguments.
if [ "$#" -gt 0 ] || [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
    supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
        sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
      echo "Usage: $(basename $0) CONTAINER_TYPE"
      echo "       CONTAINER_TYPE can be one of [${supported_container_types}]"
      exit 1
fi

# Determine the docker image name
DOCKER_IMG_NAME="ratex.${CONTAINER_TYPE}"

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Print arguments.
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMG_NAME}"
echo ""

# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build ${DOCKER_BUILD_ARGS} -t ${DOCKER_IMG_NAME} -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"
