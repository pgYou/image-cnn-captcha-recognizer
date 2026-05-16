# CNN CAPTCHA Recognizer

基于 CNN 的验证码生成、分割与识别系统（大学毕设项目）。

## 项目结构

```
├── core/                        # 核心代码
│   ├── generator.py             # 验证码生成（CaptchaGenerator）
│   ├── image.py                 # 图片封装类（captchaImg）
│   ├── preprocessor/            # 图像预处理
│   │   ├── denoiser.py          # 灰度化、二值化、降噪
│   │   ├── splitter.py          # 字符分割、滴水法处理粘连
│   │   └── tool.py              # PIL/OpenCV 格式转换工具
│   └── model/                   # CNN 模型
│       ├── network.py           # 模型结构定义
│       ├── trainModel.py        # 训练（Trainer）
│       └── identification.py    # 推理识别（Identificator）
├── scripts/                     # 可执行脚本
│   ├── train.py                 # 训练入口
│   ├── recognize.py             # 识别入口
│   └── dev.py                   # 开发调试工具
├── docs/                        # 文档
│   ├── training.md              # 训练指南
│   └── inference.md             # 推理识别指南
├── data/
│   ├── train/                   # 训练集（从 Release 下载）
│   ├── val/                     # 验证集（从 Release 下载）
│   └── samples/                 # 样例验证码图片
├── saved_model/                 # 模型权重（从 Release 下载）
└── requirements.txt
```

## 处理流程

1. **生成验证码** — `CaptchaGenerator` 使用 `captcha` 库生成 4 位小写字母验证码
2. **预处理** — 灰度化 → 二值化 → 八邻域降噪，去除噪点和干扰线
3. **分割** — 轮廓检测提取单个字符，粘连字符使用滴水算法二次分割，统一 resize 为 25×40
4. **训练/识别** — 3 层 CNN（32→64→128 通道）+ 全连接层，单字符 26 分类

## CNN 模型结构

```
Input (40, 25, 1)
  → Conv2D(32, 3×3) → MaxPool → Dropout
  → Conv2D(64, 3×3) → MaxPool → Dropout
  → Conv2D(128, 3×3) → MaxPool → Dropout
  → Flatten → Dense(1024) → Dropout
  → Dense(26)
```

## 环境要求

- **Python 3.9+**
- **TensorFlow 2.13+**（原生支持 macOS Apple Silicon）
- OpenCV、Pillow、NumPy、captcha、Matplotlib

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python scripts/train.py
```

详细说明见 [docs/training.md](docs/training.md)。

### 识别验证码

```bash
python scripts/recognize.py
python scripts/recognize.py data/samples/edgc.png
```

详细说明见 [docs/inference.md](docs/inference.md)。

### 开发调试

```bash
python scripts/dev.py prepare 10000          # 一键生成训练集+验证集
python scripts/dev.py generate 1000           # 生成无噪点验证码
python scripts/dev.py segment_batch           # 批量分割验证码
python scripts/dev.py denoise                 # 降噪流程演示
```

## 文档

- [训练指南](docs/training.md) — 环境搭建、数据准备、模型训练
- [推理识别指南](docs/inference.md) — 模型加载、验证码识别
