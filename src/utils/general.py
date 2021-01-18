'''This part is adapted from 
https://github.com/ultralytics/yolov3/blob/master/utils/general.py
for bounding boxes and landmarks.
'''

import numpy as np
import torch
import random
import cv2

from pathlib import Path


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    for i in range(boxes.shape[1]//2):
        boxes[:, i*2].clamp_(0, img_shape[1])  # x1
        boxes[:, i*2+1].clamp_(0, img_shape[0])  # y1
    

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    for i in range(coords.shape[1]//2):
        coords[:, i*2] -= pad[0]  # x padding
        coords[:, i*2+1] -= pad[1]  # y padding
    coords /= gain
    clip_coords(coords, img0_shape)
    return coords


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_landmarks(x, img, color=(0,0,255), label=None, line_thickness=None):
    # Plots one bounding box on image img
    color = color or [random.randint(0, 255) for _ in range(3)]
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1 
    for i in range(len(x)//2):
        cv2.circle(img, (x[i*2], x[i*2+1]), tl, color, 4)


def check_requirements(file='../requirements.txt'):
    # Check installed dependencies meet requirements
    import pkg_resources
    requirements = pkg_resources.parse_requirements(Path(file).open())
    requirements = [x.name + ''.join(*x.specs) if len(x.specs) else x.name for x in requirements]
    pkg_resources.require(requirements)  # DistributionNotFound or VersionConflict exception if requirements not met
