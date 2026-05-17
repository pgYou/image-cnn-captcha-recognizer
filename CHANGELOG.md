# Changelog

## [2.0.0] - 2026-05-17

### Added
- CRNN + CTC 端到端模型（`core/model/network.py`），替代 v1 分割+CNN 方案
- `core/model/train.py` CRNN 训练逻辑（tf.data 管线、数据增强、CTC Loss）
- `core/model/inference.py` 端到端推理识别（Recognizer）
- `core/model/constants.py` 共享常量（字符集、CTC 编码、图片尺寸）
- `app/` 包：导出 `recognize()` 函数，支持 SDK 调用和 CLI
- `scripts/test.py` 测试脚本，生成 Anthropic 风格 HTML 报告（含环境信息、推理耗时、错误图片 base64 内嵌）
- `scripts/report_template.html` 测试报告 HTML 模板
- 支持 3-7 位变长验证码识别
- 训练数据增加变长、扭曲、多种噪声类型

### Changed
- 架构从「分割→单字符CNN」升级为「CRNN+CTC 端到端」，无需字符分割
- 模型：3 层 CNN → 4 层 CNN + 2 层 BiLSTM + CTC
- 输入：单字符 25×40 → 整图 60×160
- 推理调用方式：`from app import recognize`
- `scripts/train.py` 改为 CRNN 训练入口，支持 `--gpu` 参数
- `core/app/` 移至项目根目录 `app/`
- 修复 `NUM_TO_CHAR` 编码偏移 bug（StringLookup 与推理映射不一致）

### Deprecated
- `core/model/trainModel.py` — v1 CNN 训练，保留但不再维护
- `core/model/identification.py` — v1 单字符推理，保留但不再维护
- `core/preprocessor/` — v1 预处理（降噪、分割），CRNN 不再需要

## [1.1.0] - 2026-05-16

### Added
- Keras `core/model/network.py` 统一模型定义，训练和推理共享
- `scripts/train.py` 训练入口
- `scripts/recognize.py` 识别入口（支持命令行传参）
- `scripts/dev.py` 开发调试工具（prepare / generate / segment_batch 等）
- `scripts/release.sh` 打包发布脚本
- `docs/training.md` 训练指南
- `docs/inference.md` 推理识别指南
- `docs/develop/tf2-migration.md` TF2 迁移记录
- `requirements.txt` 依赖管理
- `.gitignore`

### Changed
- TF1 → TF2/Keras 迁移，支持 Python 3.9+ 和 macOS Apple Silicon
- 项目结构：`src/` → `core/`，脚本拆到 `scripts/`，文档拆到 `docs/`
- 数据目录：`picture/trainPic/` → `data/train/`，`picture/verifyPic/` → `data/val/`
- 模型权重：TF1 checkpoint（3 文件）→ `.weights.h5`（1 文件）
- `captchaImg.__init__` 修复 chrList 初始化 bug 和可变默认参数
- `spliter` 中 `Image.ANTIALIAS` → `Image.Resampling.LANCZOS`
- 脚本入口从 `test.py` 函数拆分为独立脚本

### Removed
- `test.py`（功能已迁移到 scripts/）

## [1.0.0] - 2019-05-09

### Added
- 验证码生成（CaptchaGenerator）
- 图像预处理：灰度化、二值化、八邻域降噪（deNoicer）
- 字符分割：轮廓检测 + 滴水法处理粘连（spliter）
- CNN 模型：3 层卷积 + 2 层全连接，单字符 26 分类（Trainer / Identificator）
- 训练数据约 35000 张，验证集准确率 99%
- 基于 TensorFlow 1.15 + Python 3.7
