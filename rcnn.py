# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:24:31 2018

@author: prasoon
"""

import tensorflow as tf
import anchors_box as abx
import model


def rcnn_proposals(proposals, bbox_offsets, cls_prob, num_classes, tf_image_shape):
    class_max_detections = 50
    class_nms_threshold = 0.3
    total_max_detections = 50
    min_prob_threshold =  0.5
    selected_boxes, selected_probs, selected_labels = [], [], []

    for class_id in range(num_classes):
        class_prob = cls_prob[:,class_id+1]
        class_boxes = bbox_offsets[:, (class_id*4):(4*class_id + 4)]
        raw_class_objects = abx.get_box_from_deltas(proposals, class_boxes)
        class_objects = abx.clip_boxes(raw_class_objects, tf_image_shape)
        
        prob_filter = tf.greater_equal(class_prob, min_prob_threshold)
        (xmin, ymin, xmax, ymax) = tf.unstack(class_objects, axis=1)
        area_filter = tf.greater(tf.maximum(xmax-xmin,0.0)*tf.maximum(ymax-ymin,0.0), 0.0)
        object_filter = tf.logical_and(area_filter, prob_filter)
    
        class_objects = tf.boolean_mask(class_objects, object_filter)
        class_prob = tf.boolean_mask(class_prob, object_filter)        
        class_objects_tf = abx.switch_xy(class_objects)
        
        obj_selected_idx = tf.image.non_max_suppression(class_objects_tf, 
                                                        class_prob, 
                                                        class_max_detections, 
                                                        iou_threshold=class_nms_threshold)
        class_objects_tf = tf.gather(class_objects_tf, obj_selected_idx)
        class_prob = tf.gather(class_prob, obj_selected_idx)
        class_objects = abx.switch_xy(class_objects_tf)

        selected_boxes.append(class_objects)
        selected_probs.append(class_prob)
        selected_labels.append(tf.tile([class_id],[tf.shape(obj_selected_idx)[0]]))


    objects = tf.concat(selected_boxes, axis=0)
    proposal_label = tf.concat(selected_labels, axis=0)
    proposal_label_prob = tf.concat(selected_probs, axis=0)
    
    k = tf.minimum(total_max_detections, tf.shape(proposal_label_prob)[0])
    top_k = tf.nn.top_k(proposal_label_prob, k=k)
    top_k_proposal_label_prob = top_k.values
    top_k_objects = tf.gather(objects, top_k.indices)
    top_k_proposal_label = tf.gather(proposal_label, top_k.indices)

    return top_k_objects, top_k_proposal_label, top_k_proposal_label_prob 


def rcnn_loss(cls_score, proposal_cls_targets, bbox_offsets, bbox_offset_targets, num_classes):
    cls_target = tf.cast(proposal_cls_targets, tf.int32)
    cls_target_one_hot = tf.one_hot(cls_target, depth=num_classes+1)
    
    cross_entropy_per_proposal = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(cls_target_one_hot),
            logits = cls_score)
    
    #bboxes reg use non-background labels
    
    keep_ind = tf.greater(cls_target, 0)
    cls_target_keep = tf.boolean_mask(cls_target, keep_ind)
    cls_target_keep = cls_target_keep - 1
    
    cls_target_keep_one_hot = tf.one_hot(cls_target_keep, depth=num_classes)
    
    bbox_targets_keep = tf.boolean_mask(bbox_offset_targets, keep_ind)
    bbox_offsets_keep = tf.boolean_mask(bbox_offsets, keep_ind)
    
    bbox_flatten = tf.reshape(bbox_offsets_keep, [-1,4])
    cls_target_flatten = tf.reshape(cls_target_keep_one_hot,[-1])
    
    bbox_logits = tf.boolean_mask(bbox_flatten, cls_target_flatten)
    
    reg_loss_per_proposal = model.smooth_l1_loss(bbox_logits, bbox_targets_keep)
    
    rcnn_cls_loss = tf.reduce_mean(cross_entropy_per_proposal)
    rcnn_reg_loss = tf.reduce_mean(reg_loss_per_proposal)
    return rcnn_cls_loss, rcnn_reg_loss

def rcnn_preprocess(inputs):
    #inputs = inputs - MEAN 
    return inputs

def roi_pool_features(conv_features, roi_proposals,img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    roi_pool_width = 7
    roi_pool_height = 7
    
    with tf.name_scope('roi_pool'):
        x1, y1, x2, y2 = tf.unstack(roi_proposals, axis=1)
        x1 = x1 / img_shape[0]
        y1 = y1 / img_shape[1]
        x2 = x2 / img_shape[0]
        y2 = y2 / img_shape[1]
        norm_roi_props = tf.stack([y1, x1, x2, y2], axis=1)
        num_props = tf.shape(norm_roi_props)[0]
        box_ids = tf.zeros((num_props,),dtype=tf.int32)
        rois_crops = tf.image.crop_and_resize(conv_features, 
                                              norm_roi_props, box_ids, 
                                              [roi_pool_height*2, roi_pool_width*2])
        roi_pool = tf.nn.max_pool(rois_crops, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
        
    return roi_pool
        

def rcnn_targets(proposals, gt_boxes, num_classes):
    #proposals = proposal_boxes
    #gt_boxes = tf_gt_boxes
    #num_classes = num_classes
    fg_fraction = 0.25
    fg_thresh = 0.5
    bg_thresh_low = 0.0
    bg_thresh_high = 0.5
    mini_batch_size = 64

    overlaps = abx.iou_boxes_tf(proposals, gt_boxes[:,:4])
    proposals_labels_shape = tf.gather(tf.shape(proposals),[0])
    
    proposal_labels = tf.fill(dims=proposals_labels_shape, value=-1.)
    max_overlaps = tf.reduce_max(overlaps,axis=1)
    bg_low_above = tf.greater_equal(max_overlaps, bg_thresh_low)
    bg_high_below = tf.less(max_overlaps, bg_thresh_high)
    bg_low_high = tf.logical_and(bg_low_above, bg_high_below)
    
    proposal_labels = tf.where(bg_low_high, 
                               x=tf.zeros_like(proposal_labels, dtype=tf.float32), 
                               y=proposal_labels )

    # best gt box for each proposal and filter for fg threhsold
    
    best_gt_overlap = tf.argmax(overlaps, axis=1)
    proposal_labels_best_gt = tf.add(tf.gather(gt_boxes[:,4],best_gt_overlap), 1.) # 0 is the background so adding 1
    
    fg_over_thresh = tf.greater_equal(max_overlaps, fg_thresh)
    
    proposal_labels = tf.where(fg_over_thresh, x=proposal_labels_best_gt,
                               y=proposal_labels)
    #best proposal for each gt box
    best_prop_idx = tf.argmax(overlaps, axis=0)
    
    is_best_box = tf.sparse_to_dense(sparse_indices=tf.reshape(best_prop_idx,[-1]), 
                                     sparse_values=True, default_value=False,
                                     output_shape=tf.cast(proposals_labels_shape,tf.int64),
                                     validate_indices=False)
    
    best_prop_gt_label = tf.sparse_to_dense(tf.reshape(best_prop_idx,[-1]),
                                         sparse_values=gt_boxes[:,4] + 1, default_value=0.,
                                         output_shape=tf.cast(proposals_labels_shape,tf.int64),
                                         validate_indices=False)

    proposal_labels = tf.where(is_best_box, x=best_prop_gt_label, y=proposal_labels)    

    fg_ind_conds = tf.logical_or(fg_over_thresh, is_best_box)
    fg_inds = tf.where(fg_ind_conds)
    
    max_bg = int(mini_batch_size*(1-fg_fraction))
    bg_inds = tf.where(tf.equal(proposal_labels,0))
    
    shuffle_bg_inds = tf.random_shuffle(bg_inds)
    num_reduce_inds = tf.shape(bg_inds)[0] - max_bg
    remove_inds = shuffle_bg_inds[:num_reduce_inds]
    is_remove = tf.sparse_to_dense(sparse_indices=remove_inds, sparse_values = True, default_value=False,
                                         output_shape=tf.cast(proposals_labels_shape,tf.int64),
                                         validate_indices=False)
    proposal_labels = tf.where(condition=is_remove, x=tf.fill(dims=proposals_labels_shape,value=-1.),
                              y=proposal_labels)
    
    target_available = tf.greater(proposal_labels, 0)
    prop_tgt_ids = tf.where(target_available)
    # get gt box is with 
    gt_boxes_idxs = tf.gather(best_gt_overlap, prop_tgt_ids)
    
    proposal_gt_boxes = tf.gather_nd(gt_boxes[:,:4], gt_boxes_idxs)
    proposal_with_target = tf.gather_nd(proposals, prop_tgt_ids)
    
    bbox_targets_non_zero = abx.get_deltas_from_box(proposal_with_target, proposal_gt_boxes)
    bbox_targets = tf.scatter_nd(prop_tgt_ids, updates=bbox_targets_non_zero, 
                                 shape=tf.cast(tf.shape(proposals), tf.int64))
    
    return proposal_labels, bbox_targets
    


