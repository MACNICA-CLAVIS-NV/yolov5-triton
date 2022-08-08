#!/usr/bin/env bash

set -eu

source ./docker_config.sh

sudo docker build \
    --build-arg BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:${PYTORCH_TAG}" \
    -t "yolov5-triton:${PYTORCH_TAG}" \
    ./