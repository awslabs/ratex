#!/usr/bin/env bash
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
