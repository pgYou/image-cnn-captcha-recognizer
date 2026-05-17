# coding:utf-8
"""CRNN + CTC 模型定义（端到端验证码识别）"""
import tensorflow as tf

from core.model.constants import (
    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, MAX_SEQ_LENGTH,
)


def build_crnn_model():
    """构建 CRNN 模型，返回 (training_model, prediction_model)

    training_model: 接收图片+标签+长度信息，输出 CTC Loss
    prediction_model: 只接收图片，输出 40×27 概率矩阵
    """
    # --- 输入 ---
    image_input = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='image')

    # --- CNN 特征提取 ---
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(image_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    # --- 序列化：宽度作为时间轴 ---
    # 形状 (3, 40, 128) → Permute → (40, 3, 128) → Reshape → (40, 384)
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    x = tf.keras.layers.Reshape((MAX_SEQ_LENGTH, -1))(x)

    # --- RNN 序列建模 ---
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.3)
    )(x)

    # --- 输出：27 分类（26 字母 + CTC 空白） ---
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # --- 推理模型 ---
    prediction_model = tf.keras.Model(inputs=image_input, outputs=output, name='crnn_predict')

    # --- 训练模型（加入 CTC Loss） ---
    label_input = tf.keras.Input(shape=(None,), name='label', dtype=tf.int32)
    input_length = tf.keras.Input(shape=(1,), name='input_length', dtype=tf.int64)
    label_length = tf.keras.Input(shape=(1,), name='label_length', dtype=tf.int64)

    loss_out = tf.keras.layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(args[0], args[1], args[2], args[3]),
        name='ctc_loss'
    )([label_input, output, input_length, label_length])

    training_model = tf.keras.Model(
        inputs=[image_input, label_input, input_length, label_length],
        outputs=loss_out,
        name='crnn_train'
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=lambda y_true, y_pred: y_pred,
    )

    return training_model, prediction_model
