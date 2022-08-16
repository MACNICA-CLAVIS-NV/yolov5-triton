#!/usr/bin/env bash

set -eu

source ./triton_config.sh

sudo docker run \
    --gpus=1 \
    -it \
    --rm \
    --net=host \
    --runtime nvidia \
    -v ${MODEL_REPO_HOST_PATH}:${MODEL_REPO_CONTAINER_PATH} \
    "${IMAGE_NAME}:${PYTORCH_TAG}" \
    ./bin/tritonserver --model-repository=${MODEL_REPO_CONTAINER_PATH} --backend-directory=./backends --backend-config=tensorrt,coalesce-request-input=true --model-control-mode=none