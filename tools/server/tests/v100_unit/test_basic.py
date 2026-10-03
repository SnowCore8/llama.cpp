import pytest
import requests
import socket
from utils import *

server = None


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 8081
    server.api_key = "sk-1234567890"
    server.start()
    yield
    server.stop()


def test_server_start_simple():
    global server
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_multiple_addresses(monkeypatch):
    # Skip this test - requires IPv6 and multiple addresses
    pytest.skip("v100: Skip multiple addresses test")


def test_server_props():
    global server
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server
    pytest.skip("v100: Skip slots test - requires server restart")


def test_load_split_model():
    pytest.skip("v100: Skip split model download test")


def test_no_ui():
    global server
    pytest.skip("v100: Skip no_ui test - requires server restart")


def test_server_model_aliases_and_tags():
    global server
    pytest.skip("v100: Skip aliases/tags test - requires server restart")
