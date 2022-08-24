# MIT License
#
# Copyright (c) 2022 MACNICA-CLAVIS-NV
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import urllib.request
import argparse
from PIL import Image, ImageDraw, ImageFont
from yolov5_utils import *
from logging import basicConfig, getLogger, DEBUG, ERROR
from common import *

basicConfig(level=DEBUG)
# basicConfig(level=ERROR)
logger = getLogger(__name__)

DEFAULT_ENGINE_FILE = 'model.plan'
DEFAULT_IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'
TRT_LOGGER = trt.Logger()
# FONT_FILE = 'DejaVuSans.ttf'
LABEL_FILE: str = os.path.join('.', 'model_repository', 'yolov5s_trt', 'coco.txt')

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

     # Load label categories
    labels = [line.rstrip('\n') for line in open(LABEL_FILE)]

    image_base_name: Tuple[str, str] = os.path.splitext(os.path.basename(image_file))

    draw = ImageDraw.Draw(pil_image)
    # fnt = ImageFont.truetype(FONT_FILE, 32)

    detected = results[0]
    for i in range(len(detected)):
        bb = detected[i]
        x0 = bb[0]
        y0 = bb[1]
        x1 = bb[2]
        y1 = bb[3]
        score = bb[4]
        category = int(bb[5])
        label = labels[category]
        print('{}\t{}\t{}\t{}\t{}\t{}'.format(x0, y0, x1, y1, score, label))
        xy = (x0, y0, x1, y1)
        draw.rectangle(xy, outline=(0, 255, 0), width=5)
        draw.text((x0, y0 - 32), label, fill=(0, 255, 0, 128), font=None)
        pil_image.save('{}{}{}'.format(image_base_name[0], '_infer', image_base_name[1]))

    print("Done.")


if __name__ == '__main__':
    main()