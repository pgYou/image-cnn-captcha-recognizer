# coding:utf-8
"""共享常量：字符集、编码表、图片尺寸"""

# 字符集
CHAR_SET = 'abcdefghijklmnopqrstuvwxyz'
CHAR_SET_LEN = len(CHAR_SET)

# CTC 编码：0=空白标记，1-26=a-z
BLANK_TOKEN = 0
NUM_CLASSES = CHAR_SET_LEN + 1  # 27

CHAR_TO_NUM = {c: i + 1 for i, c in enumerate(CHAR_SET)}
NUM_TO_CHAR = {i + 1: c for i, c in enumerate(CHAR_SET)}

# 整张验证码图片尺寸（captcha 库默认输出 160×60）
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

# CTC 序列长度（宽度方向经过池化后的时间步数）
MAX_SEQ_LENGTH = IMAGE_WIDTH // 4  # 40
