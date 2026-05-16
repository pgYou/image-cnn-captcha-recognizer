#coding:utf-8
"""开发调试工具：生成验证码、降噪演示、批量分割等"""
import copy
import os
import random
import sys
import cv2 as cv
from PIL import Image
import numpy as np

from core.generator import cg
from core.preprocessor.denoiser import dn
from core.preprocessor.splitter import spt
from core.image import captchaImg


def denoise_demo():
    """降噪流程演示"""
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


def generate(num, save_path="data/temp/"):
    """批量生成无噪点验证码并二值化"""
    print("生成验证码中…")
    imgs = cg.Generate_train_img(num)
    print("正在保存…")
    for i, each in enumerate(imgs):
        each = dn.toBina(each)
        each.save(save_path)
        if (i + 1) % 500 == 0:
            print("保存{}/{}…".format(i + 1, num))
    print("保存全部完成！共 {} 张，保存至 {}".format(num, save_path))


def segment_demo():
    """字符分割演示"""
    img = captchaImg('aeli', cv.imread('data/samples/aeli.png'))
    img = spt.segmenter(img, debug=True)
    img.show()
    img.showChrList()
    img.save_With_ChrList("data/samples/")


def drop_fall_demo():
    """滴水法分割演示"""
    image = Image.open('data/samples/aeli/2.png')
    img1, img2 = spt.drop_fall(image)
    img1.show()
    img2.show()


def segment_batch(source_path="data/temp/", save_path="data/train/"):
    """批量分割验证码到指定目录"""
    img_list = os.listdir(source_path)
    print("共 {} 张图片待分割，来源：{}，输出：{}".format(len(img_list), source_path, save_path))
    for i, each in enumerate(img_list):
        print("处理{}:{}中…".format(i + 1, each))
        imgc = captchaImg(each.split(".")[0], cv.imread(source_path + each), [])
        imgc = spt.segmenter(imgc)
        imgc.save_ChrList(save_path)
    print("分割完成！共 {} 张，保存至 {}".format(len(img_list), save_path))


def prepare(num=10000, val_ratio=0.1):
    """
    一键生成训练集和验证集
    :param num: 生成验证码总数
    :param val_ratio: 验证集比例
    """
    temp_path = "data/temp/"
    train_path = "data/train/"
    val_path = "data/val/"

    # 生成验证码
    generate(num, temp_path)

    # 随机拆分
    img_list = os.listdir(temp_path)
    random.shuffle(img_list)
    val_count = int(len(img_list) * val_ratio)
    val_list = img_list[:val_count]
    train_list = img_list[val_count:]

    print("拆分完成：训练集 {} 张，验证集 {} 张".format(len(train_list), len(val_list)))

    # 分割训练集
    print("正在分割训练集…")
    for i, each in enumerate(train_list):
        if (i + 1) % 500 == 0:
            print("训练集 {}/{}…".format(i + 1, len(train_list)))
        imgc = captchaImg(each.split(".")[0], cv.imread(temp_path + each), [])
        imgc = spt.segmenter(imgc)
        imgc.save_ChrList(train_path)

    # 分割验证集
    print("正在分割验证集…")
    for i, each in enumerate(val_list):
        if (i + 1) % 100 == 0:
            print("验证集 {}/{}…".format(i + 1, len(val_list)))
        imgc = captchaImg(each.split(".")[0], cv.imread(temp_path + each), [])
        imgc = spt.segmenter(imgc)
        imgc.save_ChrList(val_path)

    # 清理临时文件
    for each in img_list:
        os.remove(temp_path + each)

    print("全部完成！训练集：{}，验证集：{}".format(train_path, val_path))


if __name__ == '__main__':
    commands = {
        'denoise': denoise_demo,
        'generate': lambda: generate(
            int(sys.argv[2]) if len(sys.argv) > 2 else 100,
            sys.argv[3] if len(sys.argv) > 3 else "data/temp/",
        ),
        'segment_demo': segment_demo,
        'drop_fall': drop_fall_demo,
        'segment_batch': lambda: segment_batch(
            sys.argv[2] if len(sys.argv) > 2 else "data/temp/",
            sys.argv[3] if len(sys.argv) > 3 else "data/train/",
        ),
        'prepare': lambda: prepare(
            int(sys.argv[2]) if len(sys.argv) > 2 else 10000,
            float(sys.argv[3]) if len(sys.argv) > 3 else 0.1,
        ),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("用法: python scripts/dev.py <命令> [参数]")
        print()
        print("可用命令:")
        print("  prepare [总数] [验证集比例]      一键生成训练集+验证集，默认 10000 张，10% 验证集")
        print("  generate [数量] [输出目录]        生成无噪点验证码，默认输出 data/temp/")
        print("  segment_batch [来源目录] [输出目录] 批量分割验证码，默认 data/temp/ → data/train/")
        print("  denoise                          降噪流程演示")
        print("  segment_demo                     字符分割演示")
        print("  drop_fall                        滴水法分割演示")
        sys.exit(1)

    commands[sys.argv[1]]()
