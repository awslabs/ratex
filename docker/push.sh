#!/usr/bin/env bash
set -e

#
# Push the docker image to docker hub.
#
# Usage: push.sh <CONTAINER_TYPE> <VERSION> <PASSWORD>
#
# CONTAINER_TYPE: It can be ci_gpu.
#                 The image named "razor.<CONTAINER_TYPE>:latest" will be pushed.
#
# VERSION: The pushed version. The format should be like v0.12.
#
# PASSWORD: The Meta docker hub account password.
#
DOCKER_HUB_ACCOUNT=metaprojdev

# Get the container type.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the version.
VERSION=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the docker hub account password.
PASSWORD="$1"
shift 1

LOCAL_IMAGE_NAME=razor.${CONTAINER_TYPE}:latest
REMOTE_IMAGE_NAME=${DOCKER_HUB_ACCOUNT}/razor:${CONTAINER_TYPE}-${VERSION}

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${PASSWORD}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME}
docker push ${REMOTE_IMAGE_NAME}

