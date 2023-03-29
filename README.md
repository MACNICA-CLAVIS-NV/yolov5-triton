# yolov5-triton
YOLO v5 Object Detection on Triton Inference Server

Table of Contents
-----------------
- [What does this application do?](#what-does-this-application-do)
- [Model Pipeline](#model-pipeline)
- [Prerequisites](#prerequisites)
- [Server Installation (for Jetson)](#server-installation-for-jetson)
- [Server Installation (for x86_64)](#server-installation-for-x86_64)
- [Run Server (for Jetson)](#run-server-for-jetson)
- [Run Server (for x86_64)](#run-server-for-x86_64)
- [Install Client](#install-client)
- [Run Client](#run-client)

## What does this application do?

This application demonstrates the following things.
- How to prepare TensorRT model for [NVIDIA Triton Inference Server](https://github.com/triton-inference-server)
- How to launch NVIDIA Triton Inference Server
- How to form a pipeline with the model ensemble
- How to implement client applications for Triton Inference Server

## Model Pipeline

The below pipeline is formed with [the model ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models). 

| Order | Model Name | Backend | Input<br>Type | Input<br>Dimension | Output<br>Type | Output<br>Dimension | Description |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | preprocess | Python | UINT8 | [3, 384, 640] | FP32 | [3, 384, 640] | Type Conversion<br>Normalization |
| 2 | yolov5s_trt | TensorRT | FP32 | [3, 384, 640] | FP32 | [15120, 85] | Object Detection |
| 3 | postprocess | Python | FP32 | [15120, 85] | FP32 | [1, -1, 6] | Bounding Box Generation<br>Non-Maximum Suppression |

The pipeline output [1, -1, 6] consists of 1 * N * [x0, y0, x1, y1, score, class].<br>
N : The number of the detected bounding boxes<br>
(x0, y0) : The coordinate of the top-left corner of the detected bounding box<br>
(x1, y1) : The coordinate of the bottom-right corner of the detected bounding box

## Prerequisites

### Server
- Jetson Xavier/Orin or x86_64 Linux with NVIDIA GPU
- For Jetson, JetPack 5.0.2 or later
- For x86_64, [NGC](https://catalog.ngc.nvidia.com/) account

### Client
- Linux(x86_64/ARM64) or Windows(x86_64)  
**No GPU resource needed for client**

## Server Installation (for Jetson)

1. Clone this repository

	```bash
	git clone https://github.com/MACNICA-CLAVIS-NV/yolov5-triton
	```

	```bash
	cd yolov5-triton/server
	```
	
1. Launch PyTorch container

	```bash
	./torch_it.sh
	```

1. Obtain YOLO v5 ONNX model

	```bash
	pip3 install -U \
		'protobuf<4,>=3.20.2' \
		numpy \
		onnx \
		pandas \
		PyYAML \
		tqdm \
		matplotlib \
		seaborn \
		psutil \
		gitpython \
		scipy \
		setuptools
	```

	```bash
	python3 torch2onnx.py yolov5s
	```

1. Covert ONNX model to TensorRT engine

	```bash
	/usr/src/tensorrt/bin/trtexec \
		--onnx=yolov5s.onnx \
		--saveEngine=model.plan \
		--workspace=4096 \
		--exportProfile=profile.json
	```

1. Copy TensorRT engine to model repository

	```bash
	cp model.plan ./model_repository/yolov5s_trt/1/
	```

1. Exit from PyTorch container

	```bash
	exit
	```

1. Build a docker image for Triton Inference Server

	```bash
	./triton_build.sh
	```
	
## Server Installation (for x86_64)
**Need [NGC](https://catalog.ngc.nvidia.com/) account**

1. Clone this repository

	```bash
	git clone https://github.com/MACNICA-CLAVIS-NV/yolov5-triton
	```

	```bash
	cd yolov5-triton/server
	```
	
1. Launch PyTorch container

	```bash
	./torch_it_x86.sh
	```

1. Obtain YOLO v5 ONNX model

	```bash
	pip3 install \
		protobuf \
		pandas \
		PyYAML \
		tqdm \
		matplotlib \
		seaborn
	```

	```bash
	python3 torch2onnx.py yolov5s
	```

1. Covert ONNX model to TensorRT engine

	```bash
	/usr/src/tensorrt/bin/trtexec \
		--onnx=yolov5s.onnx \
		--saveEngine=model.plan \
		--workspace=4096 \
		--exportProfile=profile.json
	```

1. Copy TensorRT engine to model repository

	```bash
	cp model.plan ./model_repository/yolov5s_trt/1/
	```

1. Exit from PyTorch container

	```bash
	exit
	```

## Run Server (for Jetson)

```bash
sudo jetson_clocks
```
```bash
./triton_start_grpc.sh
```

## Run Server (for x86_64)

```bash
./triton_start_grpc_x86.sh
```

## Install Client

**The client application does not need GPU resource. It can be deployed to Windows/Linux without GPU card. Virtual python environment like conda or venv is recommened.**

1. Clone this repository

	```bash
	git clone https://github.com/MACNICA-CLAVIS-NV/yolov5-triton
	```

	```bash
	cd yolov5-triton/client
	```

1. Install Python dependencies

	```bash
	pip install tritonclient[all] Pillow opencv-python
	```

## Run Client

### Image Input Inference

```bash
python infer_image.py [-h] [--url SERVER_URL] IMAGE_FILE
```

Example:
```bash
python infer_image.py --url localhost:8000 test.jpg
```

### Camera Input Inference

```bash
python infer_camera.py [-h] [--camera CAMERA_ID] [--width CAPTURE_WIDTH] [--height CAPTURE_HEIGHT] [--url SERVER_URL]
```

Example:
```bash
python infer_camera.py --camera 1 --width 640 --height 480 --url 192.168.XXX.XXX:8000
```
