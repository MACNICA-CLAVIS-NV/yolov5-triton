ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.12-py3
FROM ${BASE_IMAGE}

ARG RELEASE_FILE_URL="https://github.com/triton-inference-server/server/releases/download/v2.23.0/tritonserver2.23.0-jetpack5.0.tgz"
ARG TRITON_PATH=/triton

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /tmp

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libb64-0d \
        libre2-5 \
        libssl1.1 \
        rapidjson-dev \
        libopenblas-dev \
        libarchive-dev \
        zlib1g \
        python3 \
        python3-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        curl \
        jq && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade grpcio-tools numpy attrdict pillow

RUN mkdir -p ${TRITON_PATH}

WORKDIR ${TRITON_PATH}

RUN wget ${RELEASE_FILE_URL} && \
    tar xvf `basename ${RELEASE_FILE_URL}` && \
    rm -f `basename ${RELEASE_FILE_URL}`
