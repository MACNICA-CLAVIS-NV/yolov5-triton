#!/usr/bin/env bash

set -eu

source ./triton_config_x86.sh

sudo docker run \
    --gpus=all \
    -it \
    --rm \
    --shm-size 256mb \
    --network=host \
    -v ${MODEL_REPO_HOST_PATH}:${MODEL_REPO_CONTAINER_PATH} \
    "nvcr.io/nvidia/tritonserver:${TRITON_TAG}" \
    ./bin/tritonserver --model-repository=${MODEL_REPO_CONTAINER_PATH} --backend-directory=./backends --backend-config=tensorrt,coalesce-request-input=true --model-control-mode=none --allow-grpc=true --grpc-port=8000 --allow-http=false
