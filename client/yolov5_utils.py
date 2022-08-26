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
NUM_CLASSES: int = 80

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
        np.ascontiguousarray(img.transpose(2, 0, 1), dtype='float32') / 255.0 for img in cv_images
    ]
    return np.array(np_images)

def _preprocess_images_uint8(images: List[np.ndarray]) -> np.ndarray:
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

def preprocess_cv_images2(images: List[np.ndarray]) -> np.ndarray:
    cv_images: List[np.ndarray] = [_reorder_cv_image(img) for img in images]
    return _preprocess_images_uint8(cv_images)

def preprocess_pil_images2(images: List[Image.Image]) -> np.ndarray:
    cv_images: List[np.ndarray] = [_pil2cv(img) for img in images]
    return _preprocess_images_uint8(cv_images)


def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    xyxy: np.ndarray = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0  # Top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0  # Top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0  # Bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0  # Bottom right y
    return xyxy

def _narrow_down_candidates(
    pred: np.ndarray, conf_thresh: float) -> np.ndarray:
    logger.debug(pred.shape)

    candidates:np.ndarray = pred[..., 4] > conf_thresh
    logger.debug(candidates.shape)

    pred_cand:np.ndarray = pred[candidates]
    logger.debug(pred_cand.shape)

    if pred_cand.shape[0] == 0:
        return np.empty((0, 6))

    # conf = obj_conf * cls_conf
    pred_cand[:, 5:] *= pred_cand[:, 4:5]

    box = _xywh2xyxy(pred_cand[:, :4])
    conf: np.ndarray = np.amax(pred_cand[:, 5:], axis=1, keepdims=True)
    # classes: np.ndarray = np.argmax(pred_cand[:, 5:], axis=1, keepdims=True)
    classes: np.ndarray = np.argmax(pred_cand[:, 5:], axis=1).reshape(-1, 1)
    pred_tab: np.ndarray = cast(np.ndarray, np.hstack((box, conf, classes)))
    # logger.debug(pred_tab)
    logger.debug(pred_tab.shape)

    select: np.ndarray = np.where(pred_tab[:, 4] < conf_thresh)[0]
    logger.debug(select)
    pred_tab = np.delete(pred_tab, select, 0)
    # logger.debug(pred_tab)
    logger.debug(pred_tab.shape)

    return pred_tab

def _non_max_suppression(pred: np.ndarray, iou_thresh: float) -> np.ndarray:

    x_coords: np.ndarray = pred[:, 0]
    y_coords: np.ndarray = pred[:, 1]
    x2_coords: np.ndarray = pred[:, 2]
    y2_coords: np.ndarray = pred[:, 3]
    widths: np.ndarray = x2_coords - x_coords
    heights: np.ndarray = y2_coords - y_coords
    areas: np.ndarray = widths * heights
    ordered: np.ndarray = pred[:, 4].argsort()[::-1]

    keep: List[np.ndarray] = []

    while ordered.size > 0:
        i = ordered[0]
        keep.append(pred[i])

        xx1 = np.maximum(x_coords[i], x_coords[ordered[1:]])
        yy1 = np.maximum(y_coords[i], y_coords[ordered[1:]])
        xx2 = np.minimum(x2_coords[i], x2_coords[ordered[1:]])
        yy2 = np.minimum(y2_coords[i], y2_coords[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou < iou_thresh)[0]
        ordered = ordered[indexes + 1]

    filtered = np.array(keep)

    return filtered

def _clip_coords(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def _scale_coords(img0_shape, coords):
    img1_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    _clip_coords(coords, img0_shape)
    return coords

def postprocess(raw_pred: np.ndarray, img_shape: Tuple[int, int], 
    conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[np.ndarray]:
    batech_pred = np.copy(raw_pred)
    batech_pred = np.reshape(batech_pred, (-1, 15120, NUM_CLASSES + 5), order='C')
    detections: List[np.ndarray] = []
    for bi, pred in enumerate(batech_pred):
        pred_tab: np.ndarray = _narrow_down_candidates(pred, conf_thresh)
        pred_tab = _non_max_suppression(pred_tab, iou_thresh)
        if len(pred_tab) <= 0:
            continue
        pred_tab = _scale_coords(img_shape, pred_tab)
        detections.append(pred_tab)
    return detections
