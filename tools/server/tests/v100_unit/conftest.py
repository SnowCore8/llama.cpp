import os
import pytest
import subprocess
import time
import requests
from filelock import FileLock
from utils import *


# v100 配置：使用独立端口，不与 llama-server.sh 冲突
V100_PORT = 8081
V100_API_KEY = "sk-1234567890"
V100_HOST = "127.0.0.1"


@pytest.fixture(scope="session", autouse=True)
def configure_v100_server():
    # v100 tests use port 8081 (not 8080, to avoid conflict with llama-server.sh)
    os.environ["PORT"] = str(V100_PORT)
    print(f"\n=== v100 测试配置 ===")
    print(f"端口: {V100_PORT}")
    print(f"API Key: {V100_API_KEY}")
    print(f"不依赖 /opt/llama-server.sh (端口 8080)")
    print(f"===================\n")
    yield
    # Stop all servers at the end of session
    instances = set(server_instances)
    for server in instances:
        server.stop()


# Skip model downloads - v100 uses local models only
@pytest.fixture(scope="session", autouse=True)
def load_server_presets(configure_v100_server, tmp_path_factory):
    # v100 tests do not download models
    pass