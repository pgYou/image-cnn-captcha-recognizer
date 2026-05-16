# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os

import tensorflow as tf
from core.model.network import build_model, CHAR_SET, CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH


class TrainError(Exception):
    pass


class Trainer:
    def __init__(self, train_img_path, verify_img_path, char_set=None, model_save_dir='saved_model/model.weights.h5'):
        self.model_save_dir = model_save_dir
        self.train_img_path = train_img_path
        self.train_images_list = os.listdir(train_img_path)
        random.seed(time.time())
        random.shuffle(self.train_images_list)

        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)

        self.char_set = char_set or CHAR_SET
        self.char_set_len = len(self.char_set)

        # 获取图片尺寸
        label, captcha_array = self._load_image(train_img_path, self.train_images_list[0])
        self.image_height, self.image_width = captcha_array.shape

        # 模型测试
        print(">>> Start model test")
        batch_x, batch_y = self.get_batch(0, size=100)
        print(">>> input batch shape: {}".format(batch_x.shape))
        print(">>> input labels shape: {}".format(batch_y.shape))

    def _load_image(self, img_path, img_name):
        """返回图片的 array 和字符标签"""
        label = (img_name.split("_")[-1]).split(".")[0]
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file).convert("L")
        captcha_array = np.array(captcha_image)

        captcha_array[captcha_array > 100] = 255
        captcha_array[captcha_array <= 100] = 0
        return label, captcha_array

    def _char_to_vec(self, char):
        """字符转 one-hot 向量"""
        vec = np.zeros(self.char_set_len)
        vec[self.char_set.index(char)] = 1
        return vec

    def get_batch(self, n, size=500):
        """获取一批训练数据"""
        batch_x = np.zeros([size, self.image_height, self.image_width, 1])
        batch_y = np.zeros([size, self.char_set_len])

        max_batch = len(self.train_images_list) // size
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")

        n = n % max_batch
        this_batch = self.train_images_list[n * size:(n + 1) * size]

        for i, img_name in enumerate(this_batch):
            label, image_array = self._load_image(self.train_img_path, img_name)
            batch_x[i, :, :, 0] = image_array / 255
            batch_y[i, :] = self._char_to_vec(label)
        return batch_x, batch_y

    def get_verify_batch(self, size=100):
        """获取一批验证数据"""
        batch_x = np.zeros([size, self.image_height, self.image_width, 1])
        batch_y = np.zeros([size, self.char_set_len])

        verify_images = [random.choice(self.verify_images_list) for _ in range(size)]

        for i, img_name in enumerate(verify_images):
            label, image_array = self._load_image(self.verify_img_path, img_name)
            batch_x[i, :, :, 0] = image_array / 255
            batch_y[i, :] = self._char_to_vec(label)
        return batch_x, batch_y

    def train_cnn(self, max_steps=3000, batch_size=500):
        """训练 CNN 模型"""
        model = build_model(self.image_height, self.image_width, self.char_set_len)

        # 尝试加载已有模型
        try:
            model.load_weights(self.model_save_dir)
            print("         >>读取已有模型成功，将继续训练…")
        except Exception:
            print("         >>模型文件为空，将创建新模型")

        for step in range(1, max_steps + 1):
            batch_x, batch_y = self.get_batch(step - 1, size=batch_size)
            loss = model.train_on_batch(batch_x, batch_y)

            if step % 10 == 0:
                tx, ty = self.get_batch(step - 1, size=100)
                acc = model.evaluate(tx, ty, verbose=0)[1]
                print("第{}次训练 >>> [训练集] 准确率为 {} >>> loss {}".format(step, acc, loss))

                vx, vy = self.get_verify_batch(size=100)
                acc_v = model.evaluate(vx, vy, verbose=0)[1]
                print("         >>> [验证集] 准确率为 {} >>> loss {}".format(acc_v, loss))

                if acc_v > 0.99:
                    model.save_weights(self.model_save_dir)
                    print("验证集准确率达到99%，保存模型成功")
                    return

            if step % 500 == 0:
                model.save_weights(self.model_save_dir)
                print("定时保存模型成功")

        model.save_weights(self.model_save_dir)

    def recognize_captcha(self):
        """随机取一张训练集图片进行预测"""
        model = build_model(self.image_height, self.image_width, self.char_set_len)
        model.load_weights(self.model_save_dir)

        img_name = random.choice(self.train_images_list)
        label, captcha_array = self._load_image(self.train_img_path, img_name)

        plt.figure()
        plt.text(30, 3, "origin:" + label)
        plt.imshow(captcha_array)

        image = captcha_array.reshape(1, self.image_height, self.image_width, 1) / 255
        prediction = model.predict(image)
        predict_idx = np.argmax(prediction, axis=1)[0]
        predict_char = chr(int(predict_idx) + ord('a'))

        print("正确: {}  预测: {}".format(label, predict_char))
        plt.text(30, 1, 'predict:{}'.format(predict_char))
        plt.show()
