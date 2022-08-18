# yolov5-triton
YOLO v5 Object Detection on Triton Inference Server

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
**The client application does not need GPU resource. It can be deployed to Windows/Linux without GPU card. Virtual python environment like conda or venv is recommened. **

1. Clone this repository

	```bash
	git clone https://github.com/MACNICA-CLAVIS-NV/yolov5-triton
	```

	```bash
	cd yolov5-triton/server
	```

1. Install Python dependencies

	```bash
	pip3 install tritonclient[all] Pillow opencv-python
	```

## Run Client
