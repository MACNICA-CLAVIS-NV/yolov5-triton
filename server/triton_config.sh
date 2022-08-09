#!/usr/bin/env bash

set -eu

PYTORCH_TAG="r34.1.1-pth1.12-py3"
IMAGE_NAME='jetson-triton'
MODEL_REPO_HOST_PATH=${PWD}/model_repository
MODEL_REPO_CONTAINER_PATH=/model_repository