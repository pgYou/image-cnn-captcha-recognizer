#coding:utf-8
"""CNN 模型训练入口"""
from core.model.trainModel import Trainer

char_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',  'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

train_image_dir = "data/train/"
verify_image_dir = "data/val/"
model_save_dir = "saved_model/test_model.mdl"

tr = Trainer(train_image_dir, verify_image_dir, char_set, model_save_dir)
tr.train_cnn()
