#!/usr/bin/env bash

set -eu

source ./docker_config.sh

sudo docker run \
    -it \
    --rm \
    --net=host \
    --runtime nvidia \
    -v ${MODEL_REPO_HOST_PATH}:${MODEL_REPO_CONTAINER_PATH} \
    "yolov5-triton:${PYTORCH_TAG}" \
    ./bin/tritonserver --model-repository=${MODEL_REPO_CONTAINER_PATH} --backend-directory=./backends --backend-config=tensorflow,version=2 --model-control-mode=none