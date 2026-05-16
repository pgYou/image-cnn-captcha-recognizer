import os

import numpy as np
import tensorflow as tf

from core.model.network import build_model, CHAR_SET, IMAGE_HEIGHT, IMAGE_WIDTH


class Identificator:

    def __init__(self, model_path):
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            print("模型不存在…")

        self.image_height = IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH
        self.char_set = CHAR_SET
        self.model = build_model()
        self.model.load_weights(model_path)

    def identificate_capthca(self, cap_img):
        arrays = []
        for each in cap_img.chrList:
            if each.mode != "L":
                each = each.convert("L")
            arr = np.asarray(each, 'f')
            arr[arr > 100] = 255
            arr[arr <= 100] = 0
            arr = arr / 255
            arrays.append(arr.reshape(self.image_height, self.image_width, 1))

        if not arrays:
            return ''

        batch = np.array(arrays)
        predictions = self.model.predict(batch)
        indices = np.argmax(predictions, axis=1)
        return ''.join(chr(idx + ord('a')) for idx in indices)
