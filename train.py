import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict
import pickle
import anchors_box as abx
import tensorflow.contrib.slim as slim

from rcnn import roi_pool_features, rcnn_targets, rcnn_loss, rcnn_proposals

from model import resnet_v1_101_base, resnet_rcnn, base_network, get_rpn_preds, smooth_l1_loss, get_rpn_proposals, get_rpn_targets, rcnn_network
#generate_anchors, generate_anchors_reference, get_box_dim_center, get_box_from_deltas, clip_boxes
config = EasyDict(yaml.load(open('base_config.yml','r')))

dataset = pickle.load(open('../pickle_data/train_data_boxes.p','rb'))
images = dataset['images']
boxes = dataset['boxes']
debug_mode = True
num_epochs = 3
learning_rate = 1e-3
base_size = 64
aspect_ratios = [0.5,1,2,3]
scales = [0.5,1, 1.5]
num_anchors = 12
num_classes = 4
feature_map_shape = (1,14,14,128)
MEAN = [123.68, 116.78, 103.94]
anchors_stride = config.model.anchors.stride #16
anchors_stride = 16 #16
restore_model = False
is_training = True
pre_train = False
model_checkpoint_dir = './checkpoint/'

anchors_reference = abx.generate_anchors_reference(base_size, aspect_ratios, scales)
tf.set_random_seed(0)
tf.reset_default_graph()
#sess = tf.InteractiveSession()

tf_image = tf.placeholder(dtype=tf.float32,shape=[224,224,3],name='image_pl')
tf_gt_boxes  = tf.placeholder(dtype=tf.float32,shape=[None,5],name="boxes_pl")
#tf_gt_boxes = tf_box_labels[:,:4]

#tf_image = images[0]
#tf_box_labels = boxes[0]


tf_image_shape = tf.shape(tf_image)[0:2]
tf_image_std = tf.image.per_image_standardization(tf_image)
#tf_batch_images  = tf_image_std - [0.485, 0.456, 0.406] 
#tf_conv_features = resnet_v1_101_base(tf.expand_dims(tf_image_std, axis=0))

tf_conv_features = base_network(tf.expand_dims(tf_image_std, axis=0))

all_anchors = abx.generate_anchors(tf.shape(tf_conv_features), anchors_stride, anchors_reference)

#get cls prob softmax  and box scores for all anchors
rpn_cls_score_orig, rpn_bbox_score_orig = get_rpn_preds(
        tf_conv_features, num_anchors, all_anchors, tf_image_shape)

print('orig rpn cls score shape',rpn_cls_score_orig.shape.as_list())
print('orig rpn bbox score shape',rpn_bbox_score_orig.shape.as_list())


# TARGETS
rpn_cls_logits, rpn_cls_labels, rpn_bbox_score , bbox_targets = get_rpn_targets(
        rpn_cls_score_orig, rpn_bbox_score_orig, all_anchors,tf_gt_boxes[:,:4])
print('rpn cls logits shape',rpn_cls_logits.shape.as_list())
print('rpn bbox score shape',rpn_bbox_score.shape.as_list())

print('rpn cls logits shape',rpn_cls_logits.shape.as_list())
print('rpn bbox score shape',rpn_bbox_score.shape.as_list())



#LOsses

l1_sigma = 3.0

loss_reg_per_anchor = smooth_l1_loss(rpn_bbox_score,bbox_targets)

loss_cls_per_anchor = tf.nn.softmax_cross_entropy_with_logits_v2(logits=rpn_cls_logits, labels=rpn_cls_labels)
loss_rpn_cls = tf.reduce_mean(loss_cls_per_anchor)
loss_rpn_reg = tf.reduce_mean(loss_reg_per_anchor)

total_loss_rpn = loss_rpn_cls + loss_rpn_reg

## PROPOSALS
#proposal_boxes, proposal_scores 
rpn_prediction_dict = get_rpn_proposals(rpn_cls_score_orig, rpn_bbox_score_orig, all_anchors, tf_image_shape)

proposal_boxes = rpn_prediction_dict['rpn_proposals'] 
proposal_scores = rpn_prediction_dict['rpn_proposal_scores']

print('proposal box shape',proposal_boxes.shape.as_list())
print('proposal score shape',proposal_scores.shape.as_list())


# RCNN

proposals = tf.stop_gradient(proposal_boxes) 

proposal_targets, bbox_offset_targets = rcnn_targets(proposals , tf_gt_boxes, num_classes)

keep_proposals_idxs = tf.reshape(tf.greater_equal(proposal_targets, 0),[-1])

proposal_cls_targets = tf.boolean_mask(proposal_targets, keep_proposals_idxs)
bbox_offset_targets = tf.boolean_mask(bbox_offset_targets, keep_proposals_idxs)

if is_training:
    proposals = tf.boolean_mask(proposals , keep_proposals_idxs)

pool_features = roi_pool_features(tf_conv_features, proposals, tf_image_shape)

cls_score, cls_prob, bbox_offsets = rcnn_network(pool_features, num_classes)
#cls_score, cls_prob, bbox_offsets = resnet_rcnn(pool_features, num_classes)



objects, object_cls, object_probs = rcnn_proposals(proposals, bbox_offsets, 
                                                   cls_prob, num_classes, tf_image_shape)

rcnn_prediction_dict={}
rcnn_prediction_dict['objects'] = objects
rcnn_prediction_dict['object_cls'] = object_cls
rcnn_prediction_dict['object_probs'] = object_probs

prediction_dict = {'rpn_prediction': rpn_prediction_dict}
prediction_dict['classification'] = rcnn_prediction_dict


rcnn_cls_loss, rcnn_reg_loss = rcnn_loss(cls_score, proposal_cls_targets, 
                                         bbox_offsets, bbox_offset_targets, 
                                         num_classes)


total_loss_rcnn = rcnn_cls_loss + rcnn_reg_loss

total_loss_op = total_loss_rpn + total_loss_rcnn


optimizer = tf.train.AdamOptimizer(learning_rate)
global_step = tf.train.get_or_create_global_step()

train_op = optimizer.minimize(total_loss_op,global_step=global_step)
graph = tf.get_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if pre_train:
    restore_vars = [v for v in slim.get_model_variables() if 'logit' not in v.name and 'block4' not in v.name ]
    print('restoring from imagenet wts...')
    pretrain_init_fn = slim.assign_from_checkpoint_fn('../resnet/resnet_v1_101.ckpt', var_list=restore_vars)
    pretrain_init_fn(sess)

batch_image = images[0]
batch_boxes = boxes[0]

def evaluate(tensor):
    #batch_image = np.expand_dims(batch_image, axis=0)
    feed_dict = {tf_image:batch_image, tf_gt_boxes:batch_boxes}
    return sess.run(tensor,feed_dict=feed_dict)

saver = tf.train.Saver()

if restore_model:
    print('restoring model...')
    saver = tf.train.Saver(tf.trainable_variables()) 
    checkpoint_file = tf.train.latest_checkpoint(model_checkpoint_dir)
    saver.restore(sess,checkpoint_file)
    print('restored -',checkpoint_file)


if is_training:
    for epoch in range(num_epochs):
        for (batch_image,batch_boxes) in zip(images,boxes):
            #batch_image = np.expand_dims(batch_image, axis=0)
            if len(batch_boxes) == 0:
                continue
            feed_dict = {tf_image:batch_image, tf_gt_boxes:batch_boxes}
        
            ( _, loss, cls_loss, reg_loss, predict_dict) = sess.run(
                    [train_op, total_loss_op, loss_rpn_cls, loss_rpn_reg, prediction_dict],feed_dict = feed_dict)
            print('epoch: {}, loss {:0.2f}, reg loss {:0.2f}, cls loss {:.2f}'.format(epoch,loss, reg_loss,cls_loss))
    

    saver.save(sess,model_checkpoint_dir+'/model.ckpt')

h,w = batch_image.shape[:2]
batch_image = images[15]
gtb = boxes[10]
new_img = batch_image.copy()

prop_boxes, prop_scores = sess.run([proposal_boxes, proposal_scores], feed_dict = {tf_image:new_img})


plt_anchors = prop_boxes[:25]
new_img = batch_image.copy()

new_img = abx.draw_boxes(new_img,plt_anchors)
plt.imshow(new_img)

if not is_training:
    new_img = batch_image.copy()
    [predict_dict] = sess.run([prediction_dict], feed_dict = {tf_image:new_img})

predictions = pred_dict[0]
classification = predictions['classification']
classification['object_cls']
plt_anchors = classification['objects']
new_img = batch_image.copy()

new_img = abx.draw_boxes(new_img,plt_anchors)
plt.imshow(new_img)

preds = EasyDict(predict_dict)
rpn_dict = preds.rpn_prediction
for key in sorted(rpn_dict.keys()):
    print(key, rpn_dict[key].shape)
    
    
#for op in graph.get_operations():
#    print(op.name)
#sess.close()
