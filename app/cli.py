# coding:utf-8
"""CLI 工具：命令行识别验证码"""
import argparse
import sys

from app import recognize


def main():
    parser = argparse.ArgumentParser(description='验证码识别工具')
    parser.add_argument('image', nargs='?', default='data/samples/edgc/edgc.png',
                        help='验证码图片路径')
    parser.add_argument('--model', '-m', default='saved_model/model.weights.h5',
                        help='模型权重路径')
    args = parser.parse_args()

    result = recognize(args.image, model_path=args.model)
    print(result)


if __name__ == '__main__':
    main()
