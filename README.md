# yolov5-triton
YOLO v5 Object Detection on Triton Inference Server

Table of Contents
-----------------
- [Prerequisites](#prerequisites)
- [Server Installation (for Jetson)](#server-installation-for-jetson)
- [Server Installation (for x86_64)](#server-installation-for-x86_64)
- [Run Server (for Jetson)](#run-server-for-jetson)
- [Run Server (for x86_64)](#run-server-for-x86_64)
- [Install Client](#install-client)
- [Run Client](#run-client)

## Prerequisites

### Server
- Jetson Xavier/Orin or x86_64 Linux with NVIDIA GPU
- For Jetson, JetPack 5.0.2 or later
- For x86_64, [NGC](https://catalog.ngc.nvidia.com/) account

### Client
- Linux(x86_64/ARM64) or Windows(x86_64)

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
		--exportProfile=profile.json \
		--exportTimes=times.json
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
		--exportProfile=profile.json \
		--exportTimes=times.json
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
	cd yolov5-triton/server
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
python infer_image.py --url localhost:8000 zidane.jpg
```

### Camera Input Inference

```bash
python infer_camera.py [-h] [--camera CAMERA_ID] [--width CAPTURE_WIDTH] [--height CAPTURE_HEIGHT] [--url SERVER_URL]
```

Example:
```bash
python infer_camera.py --camera 1 --width 640 --height 480 --url 192.168.XXX.XXX
```
