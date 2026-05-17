# coding:utf-8
"""验证码识别入口（v2 端到端）"""
from core.app import recognize
import sys

image_path = sys.argv[1] if len(sys.argv) > 1 else 'data/samples/edgc/edgc.png'
result = recognize(image_path)
print(result)
