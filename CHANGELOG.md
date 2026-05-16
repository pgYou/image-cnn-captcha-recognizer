# Changelog

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
