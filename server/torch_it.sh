#!/usr/bin/env bash

set -eu

source ./triton_config.sh

sudo docker run \
	-it \
	--rm \
	--runtime nvidia \
	--network host \
	-v $PWD:/work \
	-w /work \
	"nvcr.io/nvidia/l4t-pytorch:${PYTORCH_TAG}"
