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
import argparse
import triton_client

SERVER_URL_DEFAULT: str = 'localhost:8000'
MODEL_NAME: str = 'yolov5s_trt'


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

    # Create Triton client
    client = triton_client.TritonClient(url=args.url)

    # parse model
    try:
        client.parse_model(model_name=MODEL_NAME)
    except triton_client.TritonClientError as e:
        print(e)
        sys.exit(-1)

    print('Model {} parsed successfully'.format(MODEL_NAME))

if __name__ == '__main__':
    main()