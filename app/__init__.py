# coding:utf-8
"""应用模块：提供可调用的识别函数"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from core.model.inference import Recognizer

_default_model_path = 'saved_model/model.weights.h5'
_recognizer = None


def recognize(image_path, model_path=None):
    """识别验证码图片，返回字符串

    Args:
        image_path: 验证码图片路径
        model_path: 模型权重路径（可选，默认 saved_model/model.weights.h5）

    Returns:
        识别出的验证码字符串
    """
    global _recognizer

    model_path = model_path or _default_model_path

    # 模型切换时重新加载
    if _recognizer is None or _recognizer.model_path != model_path:
        _recognizer = Recognizer(model_path)

    return _recognizer.recognize(image_path)
