import pytest
from openai import OpenAI, NotFoundError
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_files_api():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    file = client.files.create(file=("hello.txt", b"hello world"), purpose="assistants")
    assert file.id.startswith("file-")
    assert file.object == "file"
    assert file.bytes == 11
    assert file.filename == "hello.txt"
    assert file.purpose == "assistants"

    got = client.files.retrieve(file.id)
    assert got.id == file.id

    content = client.files.content(file.id)
    assert content.content == b"hello world"

    listing = client.files.list(purpose="assistants")
    assert any(item.id == file.id for item in listing.data)

    deleted = client.files.delete(file.id)
    assert deleted.deleted is True

    with pytest.raises(NotFoundError):
        client.files.retrieve(file.id)
