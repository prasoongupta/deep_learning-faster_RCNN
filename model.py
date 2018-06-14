# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:10:21 2018

@author: prasoon
"""

import tensorflow as tf
import anchors_box as abx
import numpy as np
import functools

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1, vgg, resnet_utils

debug_mode = True
'''
def test_debug(func):
    def wrapper(*func_args, **func_kwargs):
        print('function call ' + func.__name__ + '()')
        print(*func_args)
        retval = func(*func_args,**func_kwargs)
        print('function ' + func.__name__ + '() returns ' + repr(retval))
        for val in retval:
            print(tf.shape(val))
        return retval
    wrapper.__name__ = func.__name__
    return wrapper
'''

def resnet_v1_101_base(input_image):
    #input_image = tf.expand_dims(tf_image_std, axis=0)
    net = input_image
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
          net, end_points = resnet_v1.resnet_v1_101(input_image, 
                                                    num_classes=None, 
                                                    global_pool=False,
                                                    output_stride=16,
                                                    is_training=True)   
    return net

def resnet_rcnn(features, num_classes):                   
    resnet_arg_scope = resnet_utils.resnet_arg_scope(
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True,
            weight_decay=5e-4
        )
    with slim.arg_scope(resnet_arg_scope):
        with slim.arg_scope(
            [slim.batch_norm], is_training=True
        ):
            blocks = [
                resnet_utils.Block(
                    'block4',
                    resnet_v1.bottleneck,
                    [{
                        'depth': 2048,
                        'depth_bottleneck': 512,
                        'stride': 1
                    }] * 3
                )
            ]
            net = resnet_utils.stack_blocks_dense(features, blocks) 
    flat = tf.layers.flatten(net)
    cls_score = tf.layers.dense(flat,units=(num_classes+1))

    reg_score = tf.layers.dense(flat,units=(num_classes*4))

    cls_probs = tf.nn.softmax(cls_score)

    return cls_score, cls_probs, reg_score


def rcnn_network(pool_features, num_classes):
    with tf.variable_scope('rcnn_net',
                           initializer=tf.truncated_normal_initializer(stddev=0.01),
                           regularizer=tf.contrib.layers.l2_regularizer(1e-4)):
        net = pool_features
        
        net = tf.layers.conv2d(net,filters=512,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net,2,2)
        
        #net = tf.layers.conv2d(net,filters=1024,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)

        #net = tf.layers.conv2d(net,filters=2048,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)
        #net = tf.layers.max_pooling2d(net,2,2)
        #net = tf.layers.conv2d(net,filters=2048,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)
        
        flat = tf.layers.flatten(net)

        dense = tf.layers.dense(flat, 1024)
        dense = tf.layers.dense(dense, 512)

        cls_score = tf.layers.dense(dense,units=(num_classes+1))

        reg_score = tf.layers.dense(dense,units=(num_classes*4))

        cls_probs = tf.nn.softmax(cls_score)

    return cls_score, cls_probs, reg_score


def base_network(image):
    with tf.variable_scope('base_net',
                           initializer=tf.truncated_normal_initializer(stddev=0.01),
                           regularizer=tf.contrib.layers.l2_regularizer(1e-4)):
        net = image
        net = tf.layers.conv2d(net,filters=32,kernel_size=5,strides=(1,1),padding='same',activation=tf.nn.relu)
        #net = tf.layers.conv2d(net,filters=32,kernel_size=3,strides=(1,1),padding='same')
        net = tf.layers.max_pooling2d(net,2,2)

        net = tf.layers.conv2d(net,filters=64,kernel_size=5,strides=(1,1),padding='same',activation=tf.nn.relu)
        #net = tf.layers.conv2d(net,filters=64,kernel_size=3,strides=(1,1),padding='same')
        net = tf.layers.max_pooling2d(net,2,2)

        net = tf.layers.conv2d(net,filters=128,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)
        #net = tf.layers.conv2d(net,filters=64,kernel_size=3,strides=(1,1),padding='same')
        net = tf.layers.max_pooling2d(net,2,2)

        net = tf.layers.conv2d(net,filters=512,kernel_size=3,strides=(1,1),padding='same',activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net,2,2)
    return net



def smooth_l1_loss(bbox_prediction, bbox_target, sigma=3.0):
    """
    Return Smooth L1 Loss for bounding box prediction.
    Args:
        bbox_prediction: shape (1, H, W, num_anchors * 4)
        bbox_target:     shape (1, H, W, num_anchors * 4)
    Smooth L1 loss is defined as:
    0.5 * x^2                  if |x| < d
    abs(x) - 0.5               if |x| >= d
    Where d = 1 and x = prediction - target
    """
    sigma2 = sigma ** 2
    diff = bbox_prediction - bbox_target
    abs_diff = tf.abs(diff)
    abs_diff_lt_sigma2 = tf.less(abs_diff, 1.0 / sigma2)
    bbox_loss = tf.reduce_sum(
        tf.where(
            abs_diff_lt_sigma2,
            0.5 * sigma2 * tf.square(abs_diff),
            abs_diff - 0.5 / sigma2
        ), [1]
    )
    return bbox_loss

def get_rpn_preds(conv_features, num_anchors, all_anchors, image_shape):
    with tf.variable_scope('rpn', initializer=tf.truncated_normal_initializer(stddev=0.01),
                           regularizer=tf.contrib.layers.l2_regularizer(1e-4)):
        conv_out = tf.layers.conv2d(conv_features,filters=512,kernel_size=3,padding='same',activation=tf.nn.relu)

        rpn_cls_score = tf.layers.conv2d(conv_out,filters=num_anchors*2,kernel_size=1,name="rpn_cls")

        rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2])

        rpn_bbox_score = tf.layers.conv2d(conv_out,filters=num_anchors*4,kernel_size=1,name="rpn_bbox")
        print('rpn box shape',rpn_bbox_score.shape.as_list())
        rpn_bbox_score = tf.reshape(rpn_bbox_score,[-1,4])
        print('rpn box shape',rpn_bbox_score.shape.as_list())

    return rpn_cls_score, rpn_bbox_score

#@test_debug    
def get_rpn_proposals(rpn_cls_score, rpn_bbox_score, all_anchors, tf_image_shape):


    nms_iou_threshold = 0.90
    pre_nms_top_k, nms_max_output = 1200,200

    rpn_cls_prob = tf.nn.softmax(rpn_cls_score,name="rpn_cls_prob")
    all_scores = rpn_cls_prob[:,1]
    #get boxes (proposals_orig) based on box scores

    all_proposals = abx.get_box_from_deltas(all_anchors,rpn_bbox_score)

    num_all_proposals = tf.shape(all_scores)[0]
    
    #filter proposals based on min prob, zero or negative area or outside bounds
    min_prob_filter = tf.greater_equal(all_scores,0.2)

    xmin,ymin,xmax,ymax = tf.unstack(all_proposals, axis=1)
    zero_area_filter = tf.greater(tf.maximum(xmax-xmin,0.0) * tf.maximum(ymax-ymin,0.0),0.0)

    proposal_filter = tf.logical_and(min_prob_filter,zero_area_filter)

    unsorted_proposals = tf.boolean_mask(all_proposals, proposal_filter)
    unsorted_scores = tf.boolean_mask(all_scores, proposal_filter)

    unsorted_proposals = abx.clip_boxes(unsorted_proposals, tf_image_shape)


    num_filtered_proposals = tf.shape(unsorted_scores)[0]

    tf.summary.scalar('valid_proposals_ratio',
                      (tf.cast(num_filtered_proposals,tf.float32) /
                       tf.cast(num_all_proposals,tf.float32)))

    #sort proposals and get top k 

    k = tf.minimum(tf.shape(unsorted_scores)[0],pre_nms_top_k)
    top_k = tf.nn.top_k(unsorted_scores, k=k, sorted=True)
    sorted_top_k_proposals = tf.gather(unsorted_proposals, top_k.indices)
    sorted_top_k_scores = top_k.values

    #Non Max suppression based on iou thresh    

    nms_box_input = abx.switch_xy(sorted_top_k_proposals)
    nms_box_input.shape.as_list()
    nms_indices = tf.image.non_max_suppression(nms_box_input,
                                               sorted_top_k_scores, nms_max_output, nms_iou_threshold )

    nms_boxes = tf.gather(sorted_top_k_proposals, nms_indices)

    rpn_proposal_boxes = nms_boxes
    rpn_proposal_scores = tf.gather(sorted_top_k_scores, nms_indices)
    
    pred_dict = {'rpn_proposals':rpn_proposal_boxes, 'rpn_proposal_scores':rpn_proposal_scores}
    
    if debug_mode:
        pred_dict.update({'1_rpn_cls_prob':rpn_cls_prob,
                          '2_all_scores': all_scores,
                          '2_all_proposals': all_proposals,
                          '3_proposal_filter': proposal_filter,
                          '4_unsorted_proposals': unsorted_proposals,
                          '4_unsorted_scores': unsorted_scores,
                          '5_sorted_top_k_proposals': sorted_top_k_proposals,
                          '5_sorted_top_k_scores': sorted_top_k_scores,
                          '6_nms_indices': nms_indices
                          })
    
    return pred_dict

def get_rpn_targets(rpn_cls_score_orig, rpn_bbox_score_orig, all_anchors,tf_gt_boxes):
    rpn_tgt_fg_thresh = 0.3
    rpn_tgt_bg_thresh = 0.1
    rpn_minibatch_size = 64
    fg_min_ratio = 0.5

    #TODO Anchor filters for outside the image
    anchors = all_anchors

    rpn_labels = tf.fill(tf.gather(tf.shape(anchors),[0]),-1)
    overlaps = abx.iou_boxes_tf(tf.to_float(anchors),tf.to_float(tf_gt_boxes))
    max_overlaps = tf.reduce_max(overlaps,axis=1)

    neg_overlap = tf.less(max_overlaps, rpn_tgt_bg_thresh)
    neg_inds_all = tf.squeeze(tf.where(neg_overlap))

    choice_neg = tf.py_func(np.random.choice,(neg_inds_all,tf.cast(rpn_minibatch_size*fg_min_ratio,tf.int32),False),tf.int64)
    choice_neg = tf.py_func(np.sort,[choice_neg],tf.int64)

    neg_inds = tf.sparse_to_dense(choice_neg,tf.shape(rpn_labels,out_type=tf.int64),True,False)

    rpn_labels = tf.where(neg_inds, x=tf.zeros(tf.shape(rpn_labels), dtype=tf.int32),y=rpn_labels)

    gt_max_overlap = tf.reduce_max(overlaps, axis=0)
    gt_argmax_overlap = tf.equal(overlaps, gt_max_overlap)
    gt_argmax_overlap = tf.where(gt_argmax_overlap)[:,0]
    gt_argmax_overlap,uni_ind = tf.unique(gt_argmax_overlap)
    gt_argmax_overlap = tf.py_func(np.sort,[gt_argmax_overlap],tf.int64)
    gt_argmax_overlap_cond = tf.sparse_to_dense(gt_argmax_overlap,tf.shape(rpn_labels,out_type=tf.int64),True,False)

    rpn_labels = tf.where(gt_argmax_overlap_cond,x=tf.ones(tf.shape(rpn_labels),tf.int32), y=rpn_labels)
    pos_overlap_inds = tf.greater(max_overlaps, rpn_tgt_fg_thresh )
    rpn_labels = tf.where(pos_overlap_inds, x=tf.ones(tf.shape(rpn_labels),tf.int32),y=rpn_labels)


    #TODO anchor filters to keep inside images
    # for bbox targets
    anchor_gt_argmax_overlap = tf.argmax(overlaps, axis=1)
    gt_boxes_anchors = tf.gather(tf_gt_boxes, anchor_gt_argmax_overlap) # tie best anchor to the box

    bbox_targets = abx.get_deltas_from_box(anchors, gt_boxes_anchors)

    #select bbox_targets only if labels = 1


    fg_ind_conds = tf.equal(rpn_labels,1)

    bbox_targets = tf.where(fg_ind_conds, x=bbox_targets, y=tf.zeros_like(bbox_targets))

    labels_positive_ind = tf.equal(rpn_labels,1)
    labels_keep_ind = tf.not_equal(rpn_labels,-1)

    rpn_labels = tf.boolean_mask(rpn_labels,labels_keep_ind)
    rpn_cls_labels = tf.one_hot(rpn_labels,depth=2)
    rpn_cls_logits = tf.boolean_mask(rpn_cls_score_orig, labels_keep_ind)

    bbox_targets = tf.boolean_mask(bbox_targets, labels_positive_ind)
    rpn_bbox_score = tf.boolean_mask(rpn_bbox_score_orig, labels_positive_ind)

    return rpn_cls_logits, rpn_cls_labels, rpn_bbox_score , bbox_targets


