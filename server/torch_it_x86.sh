#!/usr/bin/env bash

set -eu

source ./triton_config_x86.sh

sudo docker run \
	-it \
	--rm \
    --gpus=all \
	--ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
	--network=host \
	-v $PWD:/work \
	-w /work \
	"nvcr.io/nvidia/pytorch:${PYTORCH_TAG}"
