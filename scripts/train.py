#coding:utf-8
"""CNN 模型训练入口"""
from core.model.trainModel import Trainer

train_image_dir = "data/train/"
verify_image_dir = "data/val/"
model_save_dir = "saved_model/model.weights.h5"

tr = Trainer(train_image_dir, verify_image_dir, model_save_dir=model_save_dir)
tr.train_cnn()
