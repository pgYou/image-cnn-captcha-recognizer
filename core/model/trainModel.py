# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from tensorflow.python.framework.errors_impl import NotFoundError

class TrainError(Exception):
    pass


class Trainer:
    def __init__(self, train_img_path, verify_img_path, char_set, model_save_dir):
        # 模型保存路径
        self.model_save_dir = model_save_dir

        #读取训练字符图片
        self.train_img_path = train_img_path
        # 打乱训练文件顺序
        self.train_images_list = os.listdir(train_img_path)
        random.seed(time.time())
        random.shuffle(self.train_images_list)

        # 验证集文件
        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image(train_img_path, self.train_images_list[0])
        captcha_shape = captcha_array.shape
        self.image_height, self.image_width = captcha_shape
        self.char_set = char_set
        self.char_set_len = len(char_set)

        # tf初始化占位符
        self.X = tf.placeholder(tf.float32, [None, captcha_shape[0] * captcha_shape[1]])  # 特征向量
        self.Y = tf.placeholder(tf.float32, [None, self.char_set_len])  # 标签
        self.keep_prob = tf.placeholder(tf.float32)  # dropout值
        self.w_alpha = 0.01
        self.b_alpha = 0.1

        # test model input and output
        print(">>> Start model test")
        batch_x, batch_y = self.get_batch(0, size=100)
        print(">>> input batch images shape: {}".format(batch_x.shape))
        print(">>> input batch labels shape: {}".format(batch_y.shape))

    def gen_captcha_text_image(self, img_path, img_name):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """
        # 标签,验证码命名格式：abcd_0_a.png
        label = (img_name.split("_")[-1]).split(".")[0]
        # 文件
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_image = captcha_image.convert("L")
        captcha_array = np.array(captcha_image)  # 向量化

        height = captcha_array.shape[0]
        width = captcha_array.shape[1]
        # matrix.flags.writeable = True

        # 如果没有指定阈值
        key = 100
        # 二值化
        for i in range(height):
            for j in range(width):
                if captcha_array[i][j] > key:
                    captcha_array[i][j] = 255
                else:
                    captcha_array[i][j] = 0

        return label, captcha_array

    def char2vec(self, chr):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        vector = np.zeros(self.char_set_len)
        idx = self.char_set.index(chr)
        vector[idx] = 1
        return vector

    def get_batch(self, n, size=500):
        """
        :param n: 当前训练的批次，用于定位训练图片集当前位置
        :param size: 每一批的大小
        :return: batch_x 图片向量信息  , batch_y 字符label向量信息
        """
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化一维的图片向量
        batch_y = np.zeros([size, self.char_set_len])  # 初始化图片标签向量

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")

        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]
        # print("{}:{}".format(s, e))

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.char2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def get_verify_batch(self, size=100):
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化
        batch_y = np.zeros([size, self.char_set_len])  # 初始化

        verify_images = []
        for i in range(size):
            verify_images.append(random.choice(self.verify_images_list))

        for i, img_name in enumerate(verify_images):
            label, image_array = self.gen_captcha_text_image(self.verify_img_path, img_name)
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.char2vec(label)  # 生成 oneHot
        return batch_x, batch_y


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

    def train_cnn(self):
        y_predict = self.model()
        print(">>> input batch predict shape: {}".format(y_predict.shape))
        print(">>> End model test")

        # 计算概率 损失(的函数)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        # 梯度下降 (优化的函数)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        # 计算准确率 (的函数)
        predict = tf.reshape(y_predict, [-1, self.char_set_len])  # 预测结果

        max_idx_p = tf.argmax(predict, 1)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.char_set_len]), 1)  # 标签

        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # 模型保存对象
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 恢复模型

            try:
                saver.restore(sess, self.model_save_dir)
                print("         >>读取已有模型成功，将继续训练…")
            # 判断捕获model文件夹中没有模型文件的错误
            except NotFoundError:
                print("         >>model文件夹为空，将创建新模型")

            step = 1
            for i in range(3000):
                batch_x, batch_y = self.get_batch(i, size=500)
                # 梯度下降训练
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                if step % 10 == 0:
                    # 基于训练集的测试
                    batch_x_test, batch_y_test = self.get_batch(i, size=100)
                    acc = sess.run(accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print("第{}次训练 >>> [训练集] 准确率为 {} >>> loss {}".format(step, acc, cost_))

                    # 基于验证集的测试
                    batch_x_verify, batch_y_verify = self.get_verify_batch(size=100)
                    acc = sess.run(accuracy,
                                   feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    print("         >>> [验证集] 准确率为 {} >>> loss {}".format(acc, cost_))
                    # 准确率达到99%后保存并停止
                    if acc > 0.99:
                        saver.save(sess, self.model_save_dir)
                        print("验证集准确率达到99%，保存模型成功")
                        break
                # 每训练500轮就保存一次
                if i % 500 == 0:
                    saver.save(sess, self.model_save_dir)
                    print("定时保存模型成功")
                step += 1
            saver.save(sess, self.model_save_dir)

    def recognize_captcha(self):
        label, captcha_array = self.gen_captcha_text_image(self.train_img_path,
                                                           random.choice(self.train_images_list))

        f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.1, "origin:" + label, ha='center', va='center', transform=ax.transAxes)
        plt.text(30, 3, "origin:" + label)
        plt.imshow(captcha_array)
        # 预测图片
        image = captcha_array.flatten() / 255

        y_predict = self.model()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.char_set_len]), 1)
            text_list = sess.run(predict, feed_dict={self.X: [image], self.keep_prob: 1.})
            predict_text = text_list[0].tolist()

        print("正确: {}  预测: {}".format(label, chr(int(predict_text)+ord('a'))))
        # 显示图片和预测结果

        plt.text(30, 1, 'predict:{}'.format(chr(int(predict_text)+ord('a'))))
        plt.show()
