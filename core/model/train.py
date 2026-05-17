# coding:utf-8
"""CRNN + CTC 训练逻辑"""
import os

import numpy as np
import tensorflow as tf

from core.model.constants import (
    CHAR_TO_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_SEQ_LENGTH,
)
from core.model.network import build_crnn_model


class CTCTrainer:
    def __init__(self, train_dir, val_dir, model_save_dir='saved_model/model.weights.h5',
                 batch_size=64):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self._char_lookup = tf.keras.layers.StringLookup(
            vocabulary=list(CHAR_TO_NUM.keys()), num_oov_indices=0
        )

    def _encode_label(self, label_str):
        """将字符串编码为整数索引列表"""
        return [CHAR_TO_NUM[c] for c in label_str]

    def _parse_fn(self, filepath):
        """从文件路径加载图片和标签"""
        # 从文件名提取标签（如 abcd.png → abcd）
        filename = tf.strings.split(filepath, os.sep)[-1]
        label_str = tf.strings.split(filename, '.')[0]

        # 加载图片
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = img / 255.0

        # 编码标签
        chars = tf.strings.unicode_split(label_str, 'UTF-8')
        label = tf.cast(self._char_lookup(chars), tf.int32)

        input_length = tf.constant([MAX_SEQ_LENGTH], dtype=tf.int64)
        label_length = tf.cast(tf.shape(label)[0:1], tf.int64)

        return img, label, input_length, label_length

    def _augment_fn(self, img, label, input_length, label_length):
        """数据增强（仅训练时）"""
        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.05)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)
        return img, label, input_length, label_length

    def _create_dataset(self, image_dir, shuffle=True, augment=False):
        """创建 tf.data.Dataset"""
        pattern = os.path.join(image_dir, '*.png')
        dataset = tf.data.Dataset.list_files(pattern, shuffle=shuffle)
        dataset = dataset.map(self._parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            dataset = dataset.map(self._augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
                [None],
                [1],
                [1],
            ),
            padding_values=(0.0, tf.cast(0, tf.int32), tf.cast(0, tf.int64), tf.cast(0, tf.int64)),
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train(self, epochs=50):
        """训练 CRNN 模型"""
        training_model, prediction_model = build_crnn_model()

        # 尝试加载已有权重
        try:
            prediction_model.load_weights(self.model_save_dir)
            print('加载已有模型成功，继续训练')
        except Exception:
            print('未找到已有模型，从头训练')

        train_ds = self._create_dataset(self.train_dir, shuffle=True, augment=True)
        val_ds = self._create_dataset(self.val_dir, shuffle=False, augment=False)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_dir,
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-5,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True,
            ),
        ]

        # 训练需要 dummy labels（CTC Loss 在模型内部计算）
        train_ds_with_dummy = train_ds.map(
            lambda img, label, il, ll: ((img, label, il, ll), tf.zeros((1,)))
        )
        val_ds_with_dummy = val_ds.map(
            lambda img, label, il, ll: ((img, label, il, ll), tf.zeros((1,)))
        )

        training_model.fit(
            train_ds_with_dummy,
            validation_data=val_ds_with_dummy,
            epochs=epochs,
            callbacks=callbacks,
        )

        prediction_model.save_weights(self.model_save_dir)
        print('模型保存至', self.model_save_dir)
        return prediction_model
