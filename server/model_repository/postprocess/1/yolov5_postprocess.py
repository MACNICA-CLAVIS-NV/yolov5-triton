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
from typing import Tuple, Optional, List, cast
from logging import getLogger

logger = getLogger(__name__)

NUM_CLASSES: int = 80

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

def postprocess(raw_pred: np.ndarray, 
    conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[np.ndarray]:
    batech_pred = np.copy(raw_pred)
    batech_pred = np.reshape(batech_pred, (-1, 15120, NUM_CLASSES + 5), order='C')
    detections: List[np.ndarray] = []
    for bi, pred in enumerate(batech_pred):
        pred_tab: np.ndarray = _narrow_down_candidates(pred, conf_thresh)
        pred_tab = _non_max_suppression(pred_tab, iou_thresh)
        if len(pred_tab) <= 0:
            continue
        detections.append(pred_tab)
    return detections
