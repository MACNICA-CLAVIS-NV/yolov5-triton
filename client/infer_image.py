#!/usr/bin/env python

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


import sys
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import triton_client
from yolov5_utils import *
from typing import Tuple, Optional, List, cast

SERVER_URL_DEFAULT: str = 'localhost:8000'
MODEL_NAME: str = 'yolov5s_trt'
LABEL_FILE: str = 'coco.txt'


def main():
    # Parse the command line parameters
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Triton YOLO v5 Demo')
    parser.add_argument('image',
        type=str, metavar='IMAGE_FILE', nargs=1,
        help='Image file')
    parser.add_argument('--url',
        type=str, default=SERVER_URL_DEFAULT, metavar='SERVER_URL',
        help='Triton Inference Server URL (Default: {})'.format(SERVER_URL_DEFAULT))
    args = parser.parse_args()

    # Load label categories
    labels = [line.rstrip('\n') for line in open(LABEL_FILE)]

    image_base_name: Tuple[str, str] = os.path.splitext(os.path.basename(args.image[0]))

    # Open and resize the input image
    pil_image: Image.Image = Image.open(args.image[0])
    input_batch: np.ndarray = preprocess_pil_images([pil_image])

    # Create Triton client
    client = triton_client.TritonClient(url=args.url)

    # parse model
    try:
        client.parse_model(model_name=MODEL_NAME)
    except triton_client.TritonClientError as e:
        print(e)
        sys.exit(-1)

    print('Model {} parsed successfully'.format(MODEL_NAME))

    client.infer(input_batch)

    outputs = client.get_results()
    # print(outputs)

    results:List[np.ndarray] = postprocess(
        outputs, (pil_image.height, pil_image.width))
    # print(results[0])

    draw = ImageDraw.Draw(pil_image)
    fnt = ImageFont.truetype("arial.ttf", 32)

    for bb in results[0]:
        # print(bb)
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
        draw.text((x0, y0 - 32), label, font=fnt, fill=(0, 255, 0, 128))
        pil_image.save('{}{}{}'.format(image_base_name[0], '_infer', image_base_name[1]))


if __name__ == '__main__':
    main()