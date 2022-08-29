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

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, cast
from logging import getLogger

logger = getLogger(__name__)

INPUT_WIDTH: int = 640
INPUT_HEIGHT: int = 384

def _pil2cv(src_image: Image.Image) -> np.ndarray:
    ''' Convert image in PIL format to image in OpenCV Mat format'''
    dst_image: np.ndarray  = np.array(src_image, dtype='uint8', order='C')
    return dst_image

def _reorder_cv_image(src_image: np.ndarray) -> np.ndarray:
    assert src_image.ndim == 3
    assert src_image.shape[2] == 3 or src_image.shape[2] == 4
    dst_image: np.ndarray = src_image.copy()
    if dst_image.shape[2] == 3:
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
    else:
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGRA2RGB)
    return dst_image

def _resize_input_image(
    image: np.ndarray, pad_value: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:

    h, w, c = image.shape
    ratio_width: float = float(INPUT_WIDTH) / float(w)
    ratio_height: float = float(INPUT_HEIGHT) / float(h)

    ratio: float = 1.0
    dst_width: int = INPUT_WIDTH
    dst_height: int = INPUT_HEIGHT
    pad_width: int = 0
    pad_height: int = 0
    if ratio_width < ratio_height:
        # Letter box
        ratio = ratio_width
        dst_width = INPUT_WIDTH
        dst_height = min(round(h * ratio), INPUT_HEIGHT)
        pad_height = INPUT_HEIGHT - dst_height
    else:
        # Pillar box
        ratio = ratio_height
        dst_width = min(round(w * ratio), INPUT_WIDTH)
        dst_height = INPUT_HEIGHT
        pad_width = INPUT_WIDTH - dst_width
    pad_top: int = pad_height // 2
    pad_bottom: int = pad_height - pad_top
    pad_left: int = pad_width // 2
    pad_right: int = pad_width - pad_left

    logger.debug('{} {} {} {} {} {}'.format(
        dst_width, dst_height, pad_top, pad_bottom, pad_left, pad_right
    ))
    
    # Resize
    rescaled_image: np.ndarray = cv2.resize(
        src=image, dsize=(dst_width, dst_height), interpolation=cv2.INTER_LINEAR)

    # Pad
    padded_image: np.ndarray = cv2.copyMakeBorder(src=rescaled_image, 
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, 
        borderType=cv2.BORDER_CONSTANT, value=pad_value)

    # Debug
    # cv2.imwrite('padded_image.png', padded_image)

    return padded_image

def _preprocess_images(images: List[np.ndarray]) -> np.ndarray:
    cv_images: List[np.ndarray] = [_resize_input_image(img) for img in images]
    np_images: List[np.ndarray] = [
        np.ascontiguousarray(img.transpose(2, 0, 1), dtype='uint8') for img in cv_images
    ]
    return np.array(np_images)

def preprocess_cv_images(images: List[np.ndarray]) -> np.ndarray:
    cv_images: List[np.ndarray] = [_reorder_cv_image(img) for img in images]
    return _preprocess_images(cv_images)

def preprocess_pil_images(images: List[Image.Image]) -> np.ndarray:
    cv_images: List[np.ndarray] = [_pil2cv(img) for img in images]
    return _preprocess_images(cv_images)
