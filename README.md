# CAPTCHA OCR Recognizer

基于 CRNN + CTC 的端到端验证码 OCR 识别系统。

## 项目结构

```
├── app/                         # 应用入口（SDK + CLI）
│   ├── __init__.py              # recognize() 函数导出
│   └── cli.py                   # 命令行工具
├── core/                        # 核心代码
│   ├── generator.py             # 验证码生成（CaptchaGenerator）
│   ├── image.py                 # 图片封装类（captchaImg）
│   ├── preprocessor/            # 图像预处理（v1 遗留）
│   │   ├── denoiser.py          # 灰度化、二值化、降噪
│   │   ├── splitter.py          # 字符分割、滴水法处理粘连
│   │   └── tool.py              # PIL/OpenCV 格式转换工具
│   └── model/                   # 模型
│       ├── network.py           # CRNN + CTC 模型定义
│       ├── train.py             # CRNN 训练逻辑
│       ├── inference.py         # 推理识别（Recognizer）
│       ├── constants.py         # 字符集、编码、图片尺寸常量
│       ├── trainModel.py        # v1 CNN 训练（已弃用）
│       └── identification.py    # v1 单字符推理（已弃用）
├── scripts/                     # 可执行脚本
│   ├── train.py                 # 训练入口
│   ├── test.py                  # 测试脚本（生成 HTML 报告）
│   ├── recognize.py             # 识别入口
│   ├── dev.py                   # 开发调试工具
│   └── report_template.html     # 测试报告 HTML 模板
├── docs/                        # 文档
├── data/
│   ├── train/                   # 训练集（从 Release 下载）
│   ├── val/                     # 验证集（从 Release 下载）
│   ├── test/                    # 测试集
│   └── samples/                 # 样例验证码图片
├── output/                      # 输出（测试报告等）
├── saved_model/                 # 模型权重（从 Release 下载）
└── requirements.txt
```

## 处理流程

1. **生成验证码** — `CaptchaGenerator` 生成 3-7 位小写字母验证码，支持多种噪声和扭曲模式
2. **端到端识别** — CRNN（CNN 特征提取 + BiLSTM 序列建模）+ CTC 解码，无需字符分割

## CRNN 模型结构

```
Input (60, 160, 1)
  → Conv2D(32) → BN → ReLU → MaxPool
  → Conv2D(64) → BN → ReLU → MaxPool
  → Conv2D(128) → BN → ReLU → MaxPool
  → Conv2D(128) → BN → ReLU → MaxPool
  → Permute → Reshape (40, 384)
  → BiLSTM(64) → BiLSTM(32)
  → Dense(27, softmax)
  → CTC Loss / CTC Greedy Decode
```

## 环境要求

- **Python 3.9+**
- **TensorFlow 2.13+**（原生支持 macOS Apple Silicon）
- Pillow、NumPy、captcha

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 识别验证码

```bash
python3 -m scripts.recognize data/samples/edgc/edgc.png
```

或在代码中调用：

```python
from app import recognize
result = recognize('captcha.png')
```

### 训练模型

```bash
python3 -m scripts.train --epochs 50 --batch-size 64
python3 -m scripts.train --epochs 50 --gpu       # 启用 GPU
```

### 测试并生成报告

```bash
python3 -m scripts.test data/test/
# 报告输出到 output/test_report.html
```

### 开发调试

```bash
python3 -m scripts.dev prepare 10000              # 一键生成训练集+验证集
python3 -m scripts.dev generate 1000 data/test/   # 生成测试验证码
python3 -m scripts.dev denoise                    # 降噪流程演示
```

## 文档

- [训练指南](docs/training.md) — 环境搭建、数据准备、模型训练
- [推理识别指南](docs/inference.md) — 模型加载、验证码识别
