#!/usr/bin/env bash

set -eu

PYTORCH_TAG="22.07-py3"
TRITON_TAG="22.07-py3"
MODEL_REPO_HOST_PATH=${PWD}/model_repository
MODEL_REPO_CONTAINER_PATH=/model_repository