# yolov5-triton
YOLO v5 Object Detection on Triton Inference Server

## Obtain YOLO v5 ONNX model

```bash
sudo docker run \
	-it \
	--rm \
	--runtime nvidia \
	--network host \
	-v $PWD:/work \
	-w /work \
	nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.12-py3
```

```bash
pip3 install \
	pandas \
	PyYAML \
	tqdm \
	matplotlib \
	seaborn
```

```bash
python3 torch2onnx.py yolov5s
```

```bash
exit
```

## Covert ONNX model to TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec \
	--onnx=yolov5s.onnx \
	--saveEngine=model.plan \
	--workspace=4096 \
	--exportProfile=profile.json
```

## Copy TensorRT engine to model repository

```bash
cp model.plan ./model_repository/yolov5s_trt/1/
```

## Set up Python environment

```bash
sudo apt update
```

```bash
sudo apt install python3-venv
```

```bash
python3 -m venv --system-site-packages .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -U pip
```

```bash
python -m pip install numpy
```

```bash
export PATH=$PATH:/usr/local/cuda/bin
```

```bash
python -m pip install 'pycuda<2021.1'
```

## Test

```bash
python yolov5_trt.py
```
