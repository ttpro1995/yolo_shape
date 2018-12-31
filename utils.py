# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.nets import vgg

from sklearn.model_selection import train_test_split
import json

import numpy as np
import cv2

# import matplotlib.pyplot as plt
import time

from config import *


def load():
    labels = json.load(open('train/labels.json'))

    N = len(labels)

    X = np.zeros((N, img_size, img_size, 3), dtype='uint8')

    y = np.zeros((N, cell_size, cell_size, 5 + nclass))
    for idx, label in enumerate(labels):
        img = cv2.imread("train/{}.png".format(idx))
        X[idx] = img
        for box in label['boxes']:
            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']
            # one-hot vector  object
            cl = [0] * len(classes)
            cl[classes[box['class']]] = 1
            #   boundary box
            x_center, y_center, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1

            # TODO

            # x_idx, y_idx = 0, 0
            x_idx, y_idx = int(x_center / img_size * cell_size), int(y_center / img_size * cell_size)

            # label matrix
            y[idx, y_idx, x_idx] = 1, x_center, y_center, w, h, *cl

    return X, y


def vgg16(inputs, is_training):
    """định nghĩa CNN
    Args:
      inputs: 5-D tensor [batch_size, width, height, 3]
    Return:
      iou: 4-D tensor [batch_size, 7, 7, 5*nbox + nclass]
    """
    #
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            #
            net = slim.repeat(inputs, 2, slim.conv2d, 16, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # TODO
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            net = slim.conv2d(net, 512, [1, 1], scope='fc6')

            net = slim.conv2d(net, 13, [1, 1], activation_fn=None, scope='fc7')
    return net

def compute_iou(boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    with tf.variable_scope(scope):
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss_layer(predicts, labels, scope='loss_layer'):
    """calculate loss function
    Args:
      predicts: 4-D tensor [batch_size, 7, 7, 5*nbox+n_class]
      labels: 4-D tensor [batch_size, 7, 7, 5+n_class]
    Return:
      loss: scalar
    """
    with tf.variable_scope(scope):
        offset = np.transpose(np.reshape(np.array(
            [np.arange(cell_size)] * cell_size * box_per_cell),
            (box_per_cell, cell_size, cell_size)), (1, 2, 0))
        offset = offset[None, :]
        offset = tf.constant(offset, dtype=tf.float32)
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))

        # 2  là confidence score
        predict_object = predicts[..., :box_per_cell]

        # 8  offset  boundary box  width height
        predict_box_offset = tf.reshape(predicts[..., box_per_cell:5 * box_per_cell],
                                        (-1, cell_size, cell_size, box_per_cell, 4))

        # class object
        predict_class = predicts[..., 5 * box_per_cell:]

        # offset => normalize  [0-1]
        predict_normalized_box = tf.stack(
            [(predict_box_offset[..., 0] + offset) / cell_size,
             (predict_box_offset[..., 1] + offset_tran) / cell_size,
             tf.square(predict_box_offset[..., 2]),
             tf.square(predict_box_offset[..., 3])], axis=-1)

        #
        true_object = labels[..., :1]
        true_box = tf.reshape(labels[..., 1:5], (-1, cell_size, cell_size, 1, 4))

        # normalize  [0-1] / img_size (224)
        true_normalized_box = tf.tile(true_box, (1, 1, 1, box_per_cell, 1)) / img_size
        true_class = labels[..., 5:]

        #  offset
        true_box_offset = tf.stack(
            [true_normalized_box[..., 0] * cell_size - offset,
             true_normalized_box[..., 1] * cell_size - offset_tran,
             tf.sqrt(true_normalized_box[..., 2]),
             tf.sqrt(true_normalized_box[..., 3])], axis=-1)

        #  iou
        predict_iou = compute_iou(true_normalized_box, predict_normalized_box)

        # mask  object
        object_mask = tf.reduce_max(predict_iou, 3, keepdims=True)

        #  metric  monitor
        iou_metric = tf.reduce_mean(
            tf.reduce_sum(object_mask, axis=[1, 2, 3]) / tf.reduce_sum(true_object, axis=[1, 2, 3]))

        object_mask = tf.cast((predict_iou >= object_mask), tf.float32) * true_object

        noobject_mask = tf.ones_like(object_mask) - object_mask

        ## TODO  classification loss

        class_delta = true_object * (predict_class - true_class)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')

        ## object loss
        object_delta = object_mask * (predict_object - predict_iou)

        ## TODO  object loss  object_delta
        # object_loss = 0
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss')

        ## TODO  noobject loss
        # noobject_delta = 0
        noobject_delta = noobject_mask * predict_object
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss')

        ## TODO  localization loss
        box_mask = tf.expand_dims(object_mask, 4)
        # box_delta = 0
        box_delta = box_mask * (predict_box_offset - true_box_offset)
        box_loss = tf.reduce_mean(tf.reduce_sum(tf.square(box_delta), axis=[1, 2, 3]), name='box_loss')

        loss = 0.5 * class_loss + object_loss + 0.1 * noobject_loss + 10 * box_loss

        return loss, iou_metric, predict_object, predict_class, predict_normalized_box