# 开发日志

## v1.1.0 — TF2/Keras 迁移，支持 macOS Apple Silicon

### 目的

原版代码基于 TensorFlow 1.15 + Python 3.7，无法在 macOS Apple Silicon (arm64) 上运行（TF 1.15 没有 arm64 wheel）。需要迁移到 TensorFlow 2.x / Keras，使项目能在当前机器上直接训练和推理。

### 实施方案

#### 1. 模型层迁移（TF1 → Keras）

**新增 `core/model/network.py`**，用 `tf.keras.Sequential` 定义 CNN 模型，训练和推理共享，消除原来 `trainModel.py` 和 `identification.py` 重复定义模型的问题。

API 映射：
| TF1 | Keras |
|---|---|
| `tf.nn.conv2d + bias_add + relu` | `Conv2D(activation='relu')` |
| `tf.nn.max_pool` | `MaxPooling2D` |
| `tf.nn.dropout(keep_prob=0.75)` | `Dropout(0.25)` |
| `tf.contrib.layers.xavier_initializer` | Keras 默认 glorot_uniform |
| `tf.nn.sigmoid_cross_entropy_with_logits` | `BinaryCrossentropy(from_logits=True)` |
| `tf.train.AdamOptimizer` | `tf.keras.optimizers.Adam` |
| `tf.Session + tf.train.Saver` | `model.train_on_batch / model.save_weights` |

模型架构不变：3 层卷积（32→64→128）+ 2 层全连接（1024→26）。

#### 2. 数据输入格式

输入从一维向量 `(batch, height*width)` 改为 4D 张量 `(batch, height, width, 1)`，符合 Keras 卷积层要求。

#### 3. 推理优化

- `Identificator` 在 `__init__` 中一次性加载模型，不再每次调用都重建计算图
- 4 个字符批量 `model.predict()`，不再逐个预测

#### 4. 项目结构调整

- 包名从 `src` 改为 `core`
- 脚本入口拆分到 `scripts/`（train.py、recognize.py、dev.py）
- 文档独立到 `docs/`（training.md、inference.md）
- 数据目录规范：`data/train/`、`data/val/`、`data/samples/`
- 模型权重目录：`saved_model/`
- 打包脚本：`scripts/release.sh`

#### 5. 兼容性修复

- `Image.ANTIALIAS` → `Image.Resampling.LANCZOS`（Pillow 10+ 已移除）
- `captchaImg.__init__` 的 `chrList` 初始化逻辑 bug 修复
- `MaxPooling2D` padding 参数大写改小写（新版 Keras 区分大小写）
- `Input` 层替代 Conv2D 的 `input_shape` 参数（消除 warning）
- 模型文件后缀 `.h5` → `.weights.h5`（新版 Keras 要求）

### 环境变化

| | v1.0 | v1.1 |
|---|---|---|
| Python | 3.7 | 3.9+ |
| TensorFlow | 1.15 | 2.13+ |
| macOS Apple Silicon | 不支持 | 支持 |
| 模型格式 | TF1 checkpoint（3 文件） | `.weights.h5`（1 文件） |

### 结果

在 macOS Apple Silicon 上完成训练：

- 训练数据：35071 张单字符图片
- 验证数据：288 张
- 训练 880 步后验证集准确率达到 **99%** 自动停止
- 模型保存为 `saved_model/model.weights.h5`
