#!/bin/bash
# 打包模型和数据到 output/ 目录，用于 GitHub Release 发布
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE=$(date +%Y%m%d)
OUTPUT="$ROOT/output"

mkdir -p "$OUTPUT"

echo "打包模型权重..."
cd "$ROOT/saved_model" && zip -r "$OUTPUT/model-$DATE.zip" . && cd "$ROOT"
echo "  → output/model-$DATE.zip"

echo "打包训练数据..."
cd "$ROOT/data/train" && zip -r "$OUTPUT/data-$DATE.zip" . && cd "$ROOT"
cd "$ROOT/data/val" && zip -ur "$OUTPUT/data-$DATE.zip" . && cd "$ROOT"
echo "  → output/data-$DATE.zip"

echo "完成！"
ls -lh "$OUTPUT"/*-$DATE.zip
