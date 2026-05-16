#coding:utf-8
"""验证码识别入口"""
import sys
import cv2 as cv
from core.image import captchaImg
from core.preprocessor.splitter import spt
from core.model.identification import Identificator

model_path = "saved_model/test_model.mdl"

# 从命令行参数获取图片路径，默认使用样例
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    code = image_path.split('/')[-1].split('.')[0]
else:
    image_path = "data/samples/edgc.png"
    code = "edgc"

img = captchaImg(code, cv.imread(image_path))
img = spt.segmenter(img)

idf = Identificator(model_path)
predict_code = idf.identificate_capthca(img)
print("真实值：{}，预测值：{}".format(img.code, predict_code))
