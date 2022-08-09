#!/usr/bin/env bash

set -eu

source ./triton_config.sh

sudo docker build \
    --build-arg BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:${PYTORCH_TAG}" \
    -t "${IMAGE_NAME}:${PYTORCH_TAG}" \
    ./