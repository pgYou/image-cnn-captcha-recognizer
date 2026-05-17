# coding:utf-8
"""端到端推理识别"""
import numpy as np
import tensorflow as tf
from PIL import Image

from core.model.constants import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_SEQ_LENGTH, NUM_TO_CHAR
from core.model.network import build_crnn_model


class Recognizer:
    def __init__(self, model_path='saved_model/model.weights.h5'):
        self.model_path = model_path
        _, self.prediction_model = build_crnn_model()
        self.prediction_model.load_weights(model_path)

    def recognize(self, image_path):
        """识别验证码图片，返回字符串"""
        image = self._preprocess(image_path)
        predictions = self.prediction_model.predict(image, verbose=0)

        # CTC greedy decode
        decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length=tf.constant([MAX_SEQ_LENGTH]),
            greedy=True,
        )[0][0]

        result = ''.join(
            NUM_TO_CHAR.get(idx, '')
            for idx in decoded.numpy()[0]
            if idx != -1
        )
        return result

    def _preprocess(self, image_path):
        """加载并预处理图片"""
        img = Image.open(image_path).convert('L')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
