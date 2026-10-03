import os
import pytest
import requests
from filelock import FileLock
from utils import *


# v100 配置：复用 /opt/llama-server.sh 的配置
V100_PORT = 8080
V100_API_KEY = "sk-1234567890"
V100_HOST = "127.0.0.1"


def is_server_running():
    """检查服务器是否已运行"""
    try:
        response = requests.get(
            f"http://{V100_HOST}:{V100_PORT}/health",
            headers={"Authorization": f"Bearer {V100_API_KEY}"},
            timeout=2
        )
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="session", autouse=True)
def configure_v100_server():
    # v100 tests use the same port and API key as /opt/llama-server.sh
    os.environ["PORT"] = str(V100_PORT)
    
    # Check if server is already running
    if is_server_running():
        print(f"\n✓ 服务器已在 {V100_HOST}:{V100_PORT} 运行，跳过启动")
    else:
        print(f"\n⚠ 服务器未运行，测试将自行启动服务器实例")


@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test (but not if using external server)
    if not is_server_running():
        instances = set(server_instances)
        for server in instances:
            server.stop()


# Skip model downloads - v100 uses local models only
@pytest.fixture(scope="session", autouse=True)
def load_server_presets(configure_v100_server, tmp_path_factory):
    # v100 tests do not download models
    pass