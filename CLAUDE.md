# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Complexity Guidelines
- Prefer existing libraries over custom implementations
- Avoid over-engineering: start with the simplest solution that works
- Ask before adding test cases or complex validation logic

## Common Commands

```bash
# 运行脚本（从项目根目录，使用 -m 方式）
python3 -m scripts.train                          # 训练模型
python3 -m scripts.recognize [image_path]          # 识别验证码
python3 -m scripts.dev prepare 10000               # 一键生成训练集+验证集
python3 -m scripts.dev generate 1000 [output_dir]  # 生成无噪点验证码
python3 -m scripts.dev segment_batch [src] [dst]   # 批量分割

# 打包发布
./scripts/release.sh                               # 输出到 output/
```

## Architecture

**CNN CAPTCHA 识别系统**，流程：生成验证码 → 预处理 → 分割 → CNN 逐字符分类。

核心模块在 `core/`，以单例模式导出（`cg`、`dn`、`spt`）：

- `core/generator.py` — 验证码生成（`CaptchaGenerator`，单例 `cg`）
- `core/image.py` — 图片数据封装（`captchaImg`，持有 `capImg`、`code`、`chrList`）
- `core/preprocessor/denoiser.py` — 灰度化、二值化、降噪（`deNoicer`，单例 `dn`）
- `core/preprocessor/splitter.py` — 轮廓检测 + 滴水法分割粘连字符（`spliter`，单例 `spt`）
- `core/model/network.py` — Keras CNN 模型定义（3 层卷积 + 2 层全连接），训练和推理共享
- `core/model/trainModel.py` — 训练逻辑（`Trainer`）
- `core/model/identification.py` — 推理识别（`Identificator`，从 `network.py` import 模型）

**关键约定**：
- CNN 识别的是**单字符**，不是整张验证码。输入前必须先分割。
- 字符图片统一 resize 为 25×40，26 分类（a-z）。
- 训练数据命名格式：`{code}_{index}_{char}.png`（如 `abcd_0_a.png`），从文件名提取标签。
- `core/model/trainModel.py` 和 `core/model/identification.py` 共享同一个 `network.py` 模型定义，改动架构时只需改一处。

## Environment

- Python 3.9+, TensorFlow 2.13+（支持 macOS Apple Silicon）
- `data/train/`、`data/val/`、`saved_model/` 通过 GitHub Release 下载，已在 `.gitignore` 中排除
