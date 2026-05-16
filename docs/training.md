# 训练指南

## 环境准备

### 1. 安装 Python 3.7

TensorFlow 1.15 仅支持 Python 3.7 及以下版本。

```bash
# 使用 conda（推荐）
conda create -n captcha python=3.7
conda activate captcha

# 或使用 pyenv
pyenv install 3.7.9
pyenv local 3.7.9
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表（详见 `requirements.txt`）：

```
tensorflow==1.15.0
opencv-python
Pillow
numpy
captcha
matplotlib
```

## 数据准备

### 方式一：使用已有数据

如果你已经下载了训练数据（从 GitHub Release 的 `train_data.zip` 和 `val_data.zip`），解压到项目根目录：

```bash
unzip train_data.zip -d data/train/
unzip val_data.zip -d data/val/
```

解压后目录结构：

```
data/train/          # 训练集，约 35000 张单字符图片
data/val/            # 验证集，约 300 张单字符图片
```

每张图片命名格式为 `{验证码}_{序号}_{字符}.png`，例如 `abcd_0_a.png` 表示验证码 `abcd` 的第 0 个字符 `a`，是一张 25×40 的单字符灰度图。

### 方式二：从头生成数据

#### 一键生成（推荐）

```bash
# 默认生成 10000 张验证码，10% 作为验证集
python scripts/dev.py prepare

# 自定义：生成 20000 张，15% 作为验证集
python scripts/dev.py prepare 20000 0.15
```

该命令自动完成：生成验证码 → 随机拆分训练集/验证集 → 分割为单字符图片 → 清理临时文件。

#### 分步执行

```bash
# 第一步：生成 1000 张无噪点验证码
python scripts/dev.py generate 1000

# 第二步：分割为单字符图片，输出到训练集目录
python scripts/dev.py segment_batch

# 也可以指定自定义目录
python scripts/dev.py generate 500 data/my_temp/
python scripts/dev.py segment_batch data/my_temp/ data/my_train/
```

## 开始训练

```bash
python scripts/train.py
```

训练参数在 `scripts/train.py` 中配置：

```python
train_image_dir = "data/train/"      # 训练集目录
verify_image_dir = "data/val/"       # 验证集目录
model_save_dir = "saved_model/test_model.mdl"  # 模型保存路径
```

### 训练过程

- 每 10 步打印一次训练集和验证集准确率
- 每 500 步自动保存模型
- 验证集准确率达到 99% 时自动停止并保存
- 总共最多训练 3000 轮

### 训练输出

模型权重保存在 `saved_model/` 目录：

```
saved_model/
├── checkpoint
├── test_model.mdl.data-00000-of-00001
├── test_model.mdl.index
└── test_model.mdl.meta
```

## 验证模型效果

训练完成后运行识别脚本验证：

```bash
python scripts/recognize.py
```
