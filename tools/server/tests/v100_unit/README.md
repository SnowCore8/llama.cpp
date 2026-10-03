# v100 Unit Tests

v100 专属测试模块，全部使用本地模型，无需下载。

## 配置

- **端口**：8080（与 `/opt/llama-server.sh` 一致）
- **API Key**：`sk-1234567890`（与 `/opt/llama-server.sh` 一致）
- **模型**：全部使用本地 `/opt/llama_gguf/` 目录
- **缓存**：无（不下载任何模型）

## 本地模型

| 预设名 | 本地模型 | 用途 |
|--------|----------|------|
| tinyllama2 | Qwen3.5-0.8B (1.5GB) | 基础聊天测试 |
| tinygemma3 | Qwen3.5-0.8B + mmproj | 视觉多模态测试 |
| bert_bge_small | bge-m3 (1.1GB) | Embedding 测试 |
| jina_reranker | bge-m3 | Reranking 测试 |
| router | 无模型 | Router 功能测试 |

## 运行测试

### 方式 1：使用已运行的服务器（推荐）

如果 `/opt/llama-server.sh` 已启动，测试会自动检测并复用：

```bash
cd /opt/projects/llama.cpp/tools/server/tests
python -m pytest v100_unit/ -v
```

### 方式 2：测试自行启动服务器

如果服务器未运行，测试会自行启动服务器实例：

```bash
cd /opt/projects/llama.cpp/tools/server/tests
./v100_unit/run_tests.sh
```

### 运行特定测试

```bash
# 运行特定测试文件
python -m pytest v100_unit/test_basic.py -v

# 运行特定测试
python -m pytest v100_unit/test_basic.py::test_server_start_simple -v
```

## 与原测试的区别

| 项目 | 原测试 | v100 测试 |
|------|--------|-----------|
| 模型来源 | HuggingFace 下载 | 本地模型 |
| 端口 | 动态分配 | 固定 8080 |
| API Key | 无 / 动态 | sk-1234567890 |
| 并行 | 支持 | 不支持 |
| 下载量 | ~62GB | 0 |
| 目录 | unit/ | v100_unit/ |
| 服务器复用 | 否 | 是（检测已运行的服务器） |

## 注意事项

1. **不要并行运行**：v100 测试使用固定端口 8080，不支持并行
2. **服务器检测**：测试启动时会检测 8080 端口是否已有服务器运行
3. **本地模型**：确保 `/opt/llama_gguf/` 下有必要的模型文件