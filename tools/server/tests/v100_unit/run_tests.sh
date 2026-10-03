#!/bin/bash
# v100 Unit Tests Runner
# 运行 v100 专属测试（使用本地模型，无需下载）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== v100 Unit Tests ==="
echo "使用本地模型，无需下载"
echo ""

# 检查本地模型
echo "检查本地模型..."
if [ ! -f "/opt/llama_gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-F16.gguf" ]; then
    echo "错误: 未找到 Qwen3.5-0.8B 模型"
    exit 1
fi

if [ ! -f "/opt/llama_gguf/bge-m3/bge-m3.gguf" ]; then
    echo "错误: 未找到 bge-m3 模型"
    exit 1
fi

echo "✓ 本地模型检查通过"
echo ""

# 运行测试
echo "运行测试..."
python -m pytest v100_unit/ -v --tb=short

echo ""
echo "=== 测试完成 ==="