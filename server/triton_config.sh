#!/usr/bin/env bash

set -eu

PYTORCH_TAG="r35.2.1-pth2.0-py3"
IMAGE_NAME='jetson-triton'
MODEL_REPO_HOST_PATH=${PWD}/model_repository
MODEL_REPO_CONTAINER_PATH=/model_repository