# -*- coding: utf-8 -*-

import tensorflow as tf
## slim là package đi kèm với tensorflow, giúp định nghĩa nhanh các loại mô hình deep learning
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.nets import vgg
## sklearn là một thư viện rất phổ biến trong ML, chúng ta chỉ sử dụng tran_test_split để chia data thành 2 tập
from sklearn.model_selection import train_test_split
import json
## thư viện tính toán trên matrix
import numpy as np
import cv2
# thư viện hiển thị biểu đồ
import matplotlib.pyplot as plt
import time

from utils import *
from config import *

# logging using meowlogtool
# pip install meowlogtool
# comment out if you don't need
import sys
from meowlogtool import log_util


def init_meow_log_tool():
    # log to console and file
    logger1 = log_util.create_logger("load.log", print_console=True)
    logger1.info("LOG_FILE")  # log using logger1
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1

############## meowlogtool end


if __name__ == "__main__":
    """
    define config at config.py
    """

    print("start")

    # define graph
    graph = tf.Graph()
    with graph.as_default():
        # None đại diện cho batch_size, giúp batch_size có thể thay đổi linh hoạt
        images = tf.placeholder("float", [None, img_size, img_size, 3], name="input")
        labels = tf.placeholder('float', [None, cell_size, cell_size, 8], name='label')
        is_training = tf.placeholder(tf.bool)

        logits = vgg16(images, is_training)
        loss, iou_metric, predict_object, predict_class, predict_normalized_box = loss_layer(logits, labels)

        # định nghĩa adam optimizer, để tối ưu hàm loss
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

    print("done define graph")

    # load data
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)

    print("done load data")

    # training
    print("start training now")
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # định nghĩa saver để lưu lại trọng số của mô hình, dùng trong test các ảnh mới
        saver = tf.train.Saver(max_to_keep=2)

        saver.restore(sess, "/data/yolo_shape_model/model_take_1/yolo-294")

        # tính toán loss, iou trên tập validation
        val_loss = []
        val_iou_ms = []
        for batch in range(len(X_test) // batch_size):
            val_X_batch = X_test[batch * batch_size:(batch + 1) * batch_size]
            val_y_batch = y_test[batch * batch_size:(batch + 1) * batch_size]
            total_val_loss, val_iou_m, val_predict_object, val_predict_class, val_predict_normalized_box = sess.run(
                [loss, iou_metric, predict_object, predict_class, predict_normalized_box],
                {images: val_X_batch, labels: val_y_batch, is_training: False})
            val_loss.append(total_val_loss)
            val_iou_ms.append(val_iou_m)

        # saver.save(sess, './model/yolo', global_step=epoch)
        print(
            'val_loss: {:.3f} - val_iou: {:.3f}'.format(
                  np.mean(val_loss),
                np.mean(val_iou_ms)))

    img_idx = 1
    result = interpret_output(val_predict_object[img_idx], val_predict_class[img_idx], val_predict_normalized_box[img_idx])
    draw_result(val_X_batch[img_idx] * 255, result)