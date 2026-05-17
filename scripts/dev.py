# coding:utf-8
"""开发调试工具：数据生成、演示等"""
import os
import random
import shutil
import sys

from core.generator import cg
from core.preprocessor.denoiser import dn
from core.preprocessor.splitter import spt
from core.image import captchaImg
import copy
import cv2 as cv
from PIL import Image


def generate(num, save_path='data/temp/'):
    """批量生成带噪声验证码（整图，v2 格式）"""
    print('生成验证码中…')
    os.makedirs(save_path, exist_ok=True)
    for i in range(num):
        mode = random.choice(['1', '2', '3', '4'])
        img = cg.GenerateCap(mode)
        img.save(save_path)
        if (i + 1) % 500 == 0:
            print('保存{}/{}…'.format(i + 1, num))
    print('保存全部完成！共 {} 张，保存至 {}'.format(num, save_path))


def prepare(num=10000, val_ratio=0.1):
    """一键生成训练集+验证集（整图+噪声，v2 格式）"""
    temp_path = 'data/temp/'
    train_path = 'data/train/'
    val_path = 'data/val/'

    os.makedirs(temp_path, exist_ok=True)

    # 生成带噪声的整图验证码
    generate(num, temp_path)

    # 随机拆分
    files = [f for f in os.listdir(temp_path) if f.endswith('.png')]
    random.shuffle(files)
    val_count = int(len(files) * val_ratio)
    val_files = set(files[:val_count])

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    for f in files:
        src = os.path.join(temp_path, f)
        dst_dir = val_path if f in val_files else train_path
        shutil.move(src, os.path.join(dst_dir, f))

    shutil.rmtree(temp_path, ignore_errors=True)

    print('拆分完成：训练集 {} 张，验证集 {} 张'.format(len(files) - val_count, val_count))
    print('训练集：{}，验证集：{}'.format(train_path, val_path))


def denoise_demo():
    """降噪流程演示（v1 功能）"""
    img = cg.GenerateCap('4')
    img.show()
    img = dn.toGray(img)
    img.show()
    img = dn.toBina(copy.copy(img))
    img.show()
    img = dn.deNoise_Bina_8Near(copy.deepcopy(img), 4, 2)
    img = dn.deNoise_Bina_8Near(copy.deepcopy(img), 5, 2)
    img = dn.deNoise_Bina_8Near(copy.deepcopy(img), 6, 3)
    img.show()


if __name__ == '__main__':
    commands = {
        'prepare': lambda: prepare(
            int(sys.argv[2]) if len(sys.argv) > 2 else 10000,
            float(sys.argv[3]) if len(sys.argv) > 3 else 0.1,
        ),
        'generate': lambda: generate(
            int(sys.argv[2]) if len(sys.argv) > 2 else 100,
            sys.argv[3] if len(sys.argv) > 3 else 'data/temp/',
        ),
        'denoise': denoise_demo,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print('用法: python3 -m scripts.dev <命令> [参数]')
        print()
        print('可用命令:')
        print('  prepare [总数] [验证集比例]   一键生成训练集+验证集（带噪声整图）')
        print('  generate [数量] [输出目录]    生成带噪声验证码')
        print('  denoise                      降噪流程演示')
        sys.exit(1)

    commands[sys.argv[1]]()
