#!/usr/bin/env bash

set -eu

PYTORCH_TAG="r35.1.0-pth1.13-py3"
IMAGE_NAME='jetson-triton'
MODEL_REPO_HOST_PATH=${PWD}/model_repository
MODEL_REPO_CONTAINER_PATH=/model_repository