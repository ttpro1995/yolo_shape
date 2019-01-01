# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.nets import vgg
import matplotlib.pyplot as plt
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


def iou(box1, box2):
    """ tính iou bằng numpy
    Args:
      box1: [center_x, center_y, w, h]
      box2: [center_x, center_y, w, h]
    Return:
      iou: iou
    """
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


def interpret_output(predict_object, predict_class, predict_normalized_box):
    # nhận lại img-size để ra không gian pixel
    predict_box = predict_normalized_box * img_size
    predict_object = np.expand_dims(predict_object, axis=-1)
    predict_class = np.expand_dims(predict_class, axis=-2)
    # xác suất ô boundary chứa class bằng boundary chứa object * xác suất có điều kiện của lớp đó mà ô vuông chứa object
    class_probs = predict_object * predict_class

    # giữ các boundary box mà có xác suất chứa lớp >= 0.2
    filter_mat_probs = np.array(class_probs >= 0.2, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = predict_box[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    class_probs_filtered = class_probs[filter_mat_probs]

    # chọn index của lớp có xác xuất lớp nhất lại mỗi boundary box
    classes_num_filtered = np.argmax(
        filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    # giữ lại boundary box dự đoán có xác xuất lớp nhất
    argsort = np.array(np.argsort(class_probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    class_probs_filtered = class_probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    # thuật toán non-maximun suppression
    for i in range(len(boxes_filtered)):
        if class_probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > 0.5:
                class_probs_filtered[j] = 0.0

    # filter bước cuối bỏ những boundary overlap theo thuật toán trên
    filter_iou = np.array(class_probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    class_probs_filtered = class_probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [classes_num_filtered[i],
             boxes_filtered[i][0],
             boxes_filtered[i][1],
             boxes_filtered[i][2],
             boxes_filtered[i][3],
             class_probs_filtered[i]])

    return result

def draw_result_with_name(img, result, name):
    plt.figure(figsize=(10, 10), dpi=40)
    img = np.pad(img, [(50, 50), (50, 50), (0, 0)], mode='constant', constant_values=255)
    for i in range(len(result)):
        x = int(result[i][1]) + 50
        y = int(result[i][2]) + 50
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (231, 76, 60), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x - w + 50, y - h), (46, 204, 113), -1)
        cv2.putText(
            img, '{} : {:.2f}'.format(result[i][0], result[i][5]),
            (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (0, 0, 0), 1, cv2.LINE_AA)

    plt.imshow(img)
    plt.imsave(name, img)
    plt.clf()

def draw_result(img, result):
    """ hiển thị kết quả dự đoán
    Args:
      img: ảnh
      result: giá trị sinh ra ở hàm trên
    """
    plt.figure(figsize=(10, 10), dpi=40)
    img = np.pad(img, [(50, 50), (50, 50), (0, 0)], mode='constant', constant_values=255)
    for i in range(len(result)):
        x = int(result[i][1]) + 50
        y = int(result[i][2]) + 50
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (231, 76, 60), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x - w + 50, y - h), (46, 204, 113), -1)
        cv2.putText(
            img, '{} : {:.2f}'.format(result[i][0], result[i][5]),
            (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (0, 0, 0), 1, cv2.LINE_AA)

    plt.imshow(img)
    plt.imsave("output.png", img)
    plt.xticks([])
    plt.yticks([])