# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:12:50 2018

@author: prasoon
"""
import tensorflow as tf
import numpy as np
import cv2


def generate_anchors(feature_map_shape, anchors_stride, anchors_reference):
    """
    Generate anchor for an image.
    Using the feature map, the output of the pretrained network for an
    image, and the anchor_reference generated using the anchor config
    values. We generate a list of anchors.

    Anchors are just fixed bounding boxes of different ratios and sizes
    that are uniformly generated throught the image.

    Args:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN. Should be (batch, height, width, depth).

    Returns:
        all_anchors: A flattened Tensor with all the anchors of shape
            `(num_anchors_per_points * feature_width * feature_height, 4)`
            using the (x1, y1, x2, y2) convention.
    """
    with tf.variable_scope('generate_anchors'):
        grid_width = feature_map_shape[2]  # width
        grid_height = feature_map_shape[1]  # height
        shift_x = tf.range(grid_width) * anchors_stride
        shift_y = tf.range(grid_height) * anchors_stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = tf.transpose(shifts)
        # Shifts now is a (H x W, 4) Tensor

        # Expand dims to use broadcasting sum.
        all_anchors = (
            np.expand_dims(anchors_reference, axis=0) +
            tf.expand_dims(shifts, axis=1)
        )

        # Flatten
        all_anchors = tf.reshape(
            all_anchors, (-1, 4)
        )
        return all_anchors
    
def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.
    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.
    Scales apply to area of object.
    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.
    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """    
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )
    return anchors

def get_box_dim_center(bboxes):
    with tf.name_scope('bboxes/dim_center'):
        bboxes = tf.cast(bboxes, tf.float32)
        x1,y1,x2,y2 = tf.split(bboxes,4,axis=1)
        h,w = y2-y1+1, x2-x1 + 1
        cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
        return w, h, cx, cy
    
def get_deltas_from_box(bboxes, gt_boxes,variances=None):
    with tf.name_scope('bboxes/box2deltas'):
        (bboxes_width, bboxes_height,
         bboxes_urx, bboxes_ury) = get_box_dim_center(bboxes)

        (gt_boxes_width, gt_boxes_height,
         gt_boxes_urx, gt_boxes_ury) = get_box_dim_center(gt_boxes)
    
        if variances is None:
            variances = [1., 1.]
    
        targets_dx = (gt_boxes_urx - bboxes_urx)/(bboxes_width * variances[0])
        targets_dy = (gt_boxes_ury - bboxes_ury)/(bboxes_height * variances[0])
    
        targets_dw = tf.log(gt_boxes_width / bboxes_width) / variances[1]
        targets_dh = tf.log(gt_boxes_height / bboxes_height) / variances[1]
    
        targets = tf.concat(
            [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)
    
    return targets

    
def get_box_from_deltas(roi, deltas, variances=None):
    with tf.name_scope('bboxes/deltas2box'):
        (roi_width, roi_height,
         roi_urx, roi_ury) = get_box_dim_center(roi)

        dx, dy, dw, dh = tf.split(deltas, 4, axis=1)

        if variances is None:
            variances = [1., 1.]

        pred_ur_x = dx * roi_width * variances[0] + roi_urx
        pred_ur_y = dy * roi_height * variances[0] + roi_ury
        pred_w = tf.exp(dw * variances[1]) * roi_width
        pred_h = tf.exp(dh * variances[1]) * roi_height

        bbox_x1 = pred_ur_x - 0.5 * pred_w
        bbox_y1 = pred_ur_y - 0.5 * pred_h

        # This -1. extra is different from reference implementation.
        bbox_x2 = pred_ur_x + 0.5 * pred_w - 1.
        bbox_y2 = pred_ur_y + 0.5 * pred_h - 1.

        bboxes = tf.concat(
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)
        return bboxes

def switch_xy(bboxes):
    with tf.name_scope('bboxes/switchxy'):
        x1,y1,x2,y2 = tf.split(bboxes,4,axis=1)
        return tf.concat([y1,x1,y2,x2],axis=1)
    
def clip_boxes(bboxes, imshape):
    with tf.name_scope('bboxes/clip_bboxes'):
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        imshape = tf.cast(imshape, dtype=tf.float32)

        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = imshape[1]
        height = imshape[0]
        x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

        y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes

def draw_boxes(img, bboxes,h=1,w=1):
    for box in bboxes:
        x1,x2 = int(box[0] * w), int(box[2] * w)
        y1,y2 = int(box[1] * h), int(box[3] * h)
        img = cv2.rectangle(img,(x1,y1), (x2,y2), color=(255,255,0))
    return img

def iou_boxes_tf(bboxes1, bboxes2):
    """
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    bboxes1 = tf.cast(bboxes1,tf.float32)
    bboxes2 = tf.cast(bboxes2,tf.float32)
    with tf.name_scope('bbox_overlap'):
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        yI1 = tf.maximum(y11, tf.transpose(y21))

        xI2 = tf.minimum(x12, tf.transpose(x22))
        yI2 = tf.minimum(y12, tf.transpose(y22))

        intersection = (
            tf.maximum(xI2 - xI1 + 1., 0.) *
            tf.maximum(yI2 - yI1 + 1., 0.)
        )

        bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - intersection

        iou = tf.maximum(intersection / union, 0)

        return iou
