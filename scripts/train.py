# coding:utf-8
"""CRNN 模型训练入口（v2 端到端）"""
import argparse
import tensorflow as tf

from core.model.train import CTCTrainer


def main():
    parser = argparse.ArgumentParser(description='CRNN 模型训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--gpu', action='store_true', help='启用 GPU 并显示设备信息')
    args = parser.parse_args()

    if args.gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('检测到 GPU:')
            for gpu in gpus:
                print(f'  {gpu.name}')
        else:
            print('未检测到 GPU，使用 CPU 训练')
    else:
        tf.config.set_visible_devices([], 'GPU')

    trainer = CTCTrainer(
        train_dir='data/train/',
        val_dir='data/val/',
        model_save_dir='saved_model/model.weights.h5',
        batch_size=args.batch_size,
    )
    trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
