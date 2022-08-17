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

import cv2
import sys
import platform
import numpy as np
import argparse

# import triton_http_client as triton_client
# import triton_grpc_client as triton_client
import triton_stream_client as triton_client

from yolov5_utils import *
from typing import Tuple, Optional, List, cast
import interval_counter


WINDOW_TITLE = 'YOLO v5 Demo'
MODEL_NAME = 'yolov5s_trt'
INFO_COLOR = (133, 15, 127)
BBOX_COLOR = (63, 255, 255)
CAMERA_ID_DEFAULT = 0
CAPTURE_WIDTH_DEFAULT = 640
CAPTURE_HEIGHT_DEFAULT = 480
SERVER_URL_DEFAULT = 'localhost:8000'
LABEL_FILE = 'coco.txt'


def draw_info(frame, interval):

    height, width, color = frame.shape
    frame_info = 'Size:{}x{}'.format(width, height)
    cv2.putText(
        frame, frame_info, (width - 280, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 1, INFO_COLOR, 2, cv2.LINE_AA
    )

    if interval is not None:
        fps = 1.0 / interval
        fpsInfo = '{0}{1:.2f}'.format('FPS:', fps)
        cv2.putText(
            frame, fpsInfo, (32, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 1, INFO_COLOR, 2, cv2.LINE_AA
        )


def draw_bboxes(image, results: List[np.ndarray], labels: List[str], batch_num: int = 0):
    img_height, img_width, _ = image.shape
    detected = results[batch_num]
    for i in range(len(detected)):
        bb = detected[i]
        x0 = bb[0]
        y0 = bb[1]
        x1 = bb[2]
        y1 = bb[3]
        score = bb[4]
        category = int(bb[5])
        label = labels[category]
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(img_width, np.floor(x1 + 0.5).astype(int))
        bottom = min(img_height, np.floor(y1 + 0.5).astype(int))
        cv2.rectangle(image,
            (left, top), (right, bottom), BBOX_COLOR, 3)
        info = '{0} {1:.2f}'.format(label, score)
        cv2.putText(image, info, (left, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1, BBOX_COLOR, 1, cv2.LINE_AA)
        print(info)


def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='Triton YOLO v5 Demo')
    parser.add_argument('--camera',
        type=int, default=CAMERA_ID_DEFAULT, metavar='CAMERA_ID',
        help='Camera ID (Default: {})'.format(CAMERA_ID_DEFAULT))
    parser.add_argument('--width',
        type=int, default=CAPTURE_WIDTH_DEFAULT, metavar='CAPTURE_WIDTH',
        help='Capture Width (Default: {})'.format(CAPTURE_WIDTH_DEFAULT))
    parser.add_argument('--height',
        type=int, default=CAPTURE_HEIGHT_DEFAULT, metavar='CAPTURE_HEIGHT',
        help='Capture Height (Default: {})'.format(CAPTURE_HEIGHT_DEFAULT))
    parser.add_argument('--url',
        type=str, default=SERVER_URL_DEFAULT, metavar='SERVER_URL',
        help='Triton Inference Server URL (Default: {})'.format(SERVER_URL_DEFAULT)
    )
    args = parser.parse_args()

    # Create Triton client
    client = triton_client.TritonClient(url=args.url)

    # Load label categories
    labels: List[str] = [line.rstrip('\n') for line in open(LABEL_FILE)]

    # Parse model
    try:
        client.parse_model(model_name=MODEL_NAME)
    except triton_client.TritonClientError as e:
        print(e)
        sys.exit(-1)

    print('Model {} parsed successfully'.format(MODEL_NAME))

    # Initialize camera device
    cam_id = args.camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Create interval counter to measure FPS
    fps_counter = interval_counter.IntervalCounter(10)

    # Define the function to detect window close event
    if platform.uname().machine == 'aarch64':
        was_window_closed = lambda: cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0
    else:
        was_window_closed = lambda: cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1

    while True:
        # Capture frame n
        ret, frame = cap.read()

        # Preprocess frame n for model spec
        target_image: np.ndarray = preprocess_pil_images([frame])

        # Get inference results for frame n-1
        outputs = client.get_results()

        # Get interval value
        interval = fps_counter.measure()

        # Submit inference request for frame n
        client.infer(target_image)

        # Postprocess frame n-1 and show bounding-box for frame n-1
        if outputs is not None:
            height, width, _ = frame.shape

            results:List[np.ndarray] = postprocess(
                outputs, (height, width))

            if len(results) > 0:
                draw_bboxes(frame, results, labels)

        draw_info(frame, interval)

        # Show captured frame n
        cv2.imshow(WINDOW_TITLE, frame)

        # Check if ESC pressed
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        # Check if the window was closed
        if was_window_closed():
            break

        # # Submit inference request for frame n
        # client.infer(target_image)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
