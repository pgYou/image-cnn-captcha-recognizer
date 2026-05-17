# coding:utf-8
"""CRNN 模型训练入口（v2 端到端）"""
from core.model.train import CTCTrainer

trainer = CTCTrainer(
    train_dir='data/train/',
    val_dir='data/val/',
    model_save_dir='saved_model/model.weights.h5',
    batch_size=64,
)
trainer.train(epochs=50)
