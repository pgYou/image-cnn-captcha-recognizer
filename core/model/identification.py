import os

import numpy as np
from PIL import Image
import tensorflow as tf

class Identificator:

    def __init__(self, model_path):
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            print("模型不存在…")

        self.char_set =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
             'i', 'j', 'k', 'l','m', 'n', 'o', 'p',
             'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
             'y', 'z']

        self.image_height, self.image_width = 40, 25
        self.char_set_len = len(self.char_set)
        self.X = tf.placeholder(tf.float32, [None, 25*40])  # 特征向量
        self.Y = tf.placeholder(tf.float32, [None, self.char_set_len])  # 标签

        self.keep_prob = tf.placeholder(tf.float32)  # dropout值
        self.w_alpha = 0.01
        self.b_alpha = 0.1

    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 1])
        print(">>> input x: {}".format(x))

        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2
        wout = tf.get_variable('name', shape=[1024, self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.char_set_len]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict

    def identificate_capthca(self, cap_img):

        captcha_arrays = []

        for each in cap_img.chrList:
            if each.mode != "L":
                each = each.convert("L")
            each = np.asarray(each, 'i')
            height = each.shape[0]
            width = each.shape[1]
            key = 100
            # 二值化
            for i in range(height):
                for j in range(width):
                    if each[i][j] > key:
                        each[i][j] = 255
                    else:
                        each[i][j] = 0
            each = each.flatten() / 255
            captcha_arrays.append(each)

        y_predict = self.model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.char_set_len]), 1)
            label = ''
            for each in captcha_arrays:
                predict_val = sess.run(predict, feed_dict={self.X: [each], self.keep_prob: 1.})
                predict_val = predict_val[0].tolist()

                label += chr(int(predict_val) + ord('a'))

        return label
