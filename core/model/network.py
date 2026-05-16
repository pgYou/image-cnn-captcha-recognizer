# coding:utf-8
"""CNN 模型定义（Keras）"""
import tensorflow as tf

# 字符集：a-z
CHAR_SET = [chr(i + ord('a')) for i in range(26)]
CHAR_SET_LEN = len(CHAR_SET)

# 单个字符图片尺寸
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 25


def build_model(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
                char_set_len=CHAR_SET_LEN):
    """构建 CNN 模型：3 层卷积 + 2 层全连接"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_height, image_width, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(char_set_len),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model
