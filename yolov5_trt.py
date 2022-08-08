import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import urllib.request
import argparse
from PIL import Image
from yolov5_utils import *
from logging import basicConfig, getLogger, DEBUG, ERROR
from common import *

basicConfig(level=DEBUG)
# basicConfig(level=ERROR)
logger = getLogger(__name__)

DEFAULT_ENGINE_FILE = 'model.plan'
DEFAULT_IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'
TRT_LOGGER = trt.Logger()

def main():

    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='YOLO v5 Demo')
    parser.add_argument('--engine',
        type=str, default=DEFAULT_ENGINE_FILE, metavar='ENGINE',
        help='TensorRT engine file (Default: {})'.format(DEFAULT_ENGINE_FILE)
    )
    parser.add_argument('--url',
        type=str, default=DEFAULT_IMAGE_URL, metavar='IMAGE_URL',
        help='Image URL (Default: {})'.format(DEFAULT_IMAGE_URL)
    )
    args = parser.parse_args()

    image_file = os.path.basename(args.url)
    if not os.path.exists(image_file):
        urllib.request.urlretrieve(args.url, image_file)

    pil_image: Image.Image = Image.open(image_file)

    input_batch: np.ndarray = preprocess_pil_images([pil_image])
    logger.debug(input_batch.shape)

    engine_file = args.engine
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = input_batch
        trt_outputs = do_inference_v2(
            context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    results:List[np.ndarray] = postprocess(
        trt_outputs[0], (pil_image.height, pil_image.width))
    logger.debug(results)

    print("Done.")


if __name__ == '__main__':
    main()