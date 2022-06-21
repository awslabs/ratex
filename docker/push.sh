#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Push the docker image to docker hub.
# The docker image will have the following tags:
#  1. The user given tag.
#  2. The timestamp tag.
#
# Usage: push.sh <CONTAINER_TYPE> <VERSION>
#
# CONTAINER_TYPE: It can be ci_gpu.
#                 The local image named "ratex.<CONTAINER_TYPE>:latest" will be pushed.
#
# TAG: The user tag, such as "latest".
#
DOCKER_HUB_ACCOUNT=metaprojdev

if [ -z $DOCKER_HUB_PASSWORD ]; then
    echo "DOCKER_HUB_PASSWORD is not set"
    exit 1
fi;

# Get the container type.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the tag.
TAG=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the timestamp.
TIMESTAMP=$( echo "`date +%Y%m%d_%H%M%S`" | tr '[:upper:]' '[:lower:]' )

LOCAL_IMAGE_NAME=ratex.${CONTAINER_TYPE}:latest
REMOTE_IMAGE_TIME_NAME=${DOCKER_HUB_ACCOUNT}/ratex:${CONTAINER_TYPE}-${TIMESTAMP}
REMOTE_IMAGE_TAG_NAME=${DOCKER_HUB_ACCOUNT}/ratex:${CONTAINER_TYPE}-${TAG}

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${DOCKER_HUB_PASSWORD}

echo "Pushing ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_TIME_NAME} and ${REMOTE_IMAGE_TAG_NAME}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_TIME_NAME}
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_TAG_NAME}
docker push ${REMOTE_IMAGE_TIME_NAME}
docker push ${REMOTE_IMAGE_TAG_NAME}
