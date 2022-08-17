# yolov5-triton
YOLO v5 Object Detection on Triton Inference Server

## Installation (Server)

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

## Run Triton Inference Server

```bash
./triton_start.sh
```
