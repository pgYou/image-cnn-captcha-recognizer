# 使用模型推理识别

## 环境准备

### 1. 安装 Python 3.9+

```bash
# macOS (Homebrew)
brew install python@3.11

# 或使用 conda
conda create -n captcha python=3.11
conda activate captcha
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型文件

从 GitHub Release 下载 `model-YYYYMMDD.zip`，解压到项目根目录：

```bash
unzip model-YYYYMMDD.zip -d saved_model/
```

解压后 `saved_model/` 目录应包含：

```
saved_model/
└── model.h5
```

## 命令行识别

```bash
# 使用默认样例图片
python scripts/recognize.py

# 指定图片路径
python scripts/recognize.py data/samples/edgc.png
```

输出示例：

```
真实值：edgc，预测值：edgc
```

## 在代码中使用

```python
import cv2 as cv
from core.image import captchaImg
from core.preprocessor.splitter import spt
from core.model.identification import Identificator

# 加载模型（只需加载一次）
idf = Identificator('saved_model/model.h5')

# 读取验证码图片并分割
img = captchaImg('edgc', cv.imread('data/samples/edgc.png'))
img = spt.segmenter(img)

# 识别
result = idf.identificate_capthca(img)
print(result)  # edgc
```

## 识别流程

```
验证码图片 → 灰度化 → 二值化 → 轮廓检测分割字符 → 滴水法处理粘连 → resize 25×40 → CNN 逐字符分类 → 拼接结果
```

1. **分割**：通过轮廓检测提取单个字符，粘连字符使用滴水算法二次分割
2. **分类**：每个 25×40 的字符图片送入 CNN 模型，输出 26 分类（a-z）
3. **拼接**：将每个字符的预测结果拼接为 4 位验证码字符串
