import pytest
from openai import OpenAI, NotFoundError
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_conversations_crud():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    conv = client.conversations.create(metadata={"k": "v"}, items=[{"role": "user", "content": "hi"}])
    assert conv.id.startswith("conv_")
    assert conv.metadata == {"k": "v"}

    items = client.conversations.items.list(conv.id)
    assert len(items.data) == 1
    assert items.data[0].role == "user"

    added = client.conversations.items.create(conv.id, items=[{"role": "assistant", "content": "hello"}])
    assert len(added.data) == 1
    item = client.conversations.items.retrieve(added.data[0].id, conversation_id=conv.id)
    assert item.role == "assistant"

    remaining = client.conversations.items.delete(added.data[0].id, conversation_id=conv.id)
    assert remaining.id == conv.id

    updated = client.conversations.update(conv.id, metadata={"k": "v2"})
    assert updated.metadata == {"k": "v2"}

    deleted = client.conversations.delete(conv.id)
    assert deleted.deleted is True
    with pytest.raises(NotFoundError):
        client.conversations.retrieve(conv.id)


def test_conversation_in_responses():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    conv = client.conversations.create(items=[{"role": "user", "content": "Book"}])
    res = client.responses.create(
        model="gpt-4.1",
        input="What is the best book",
        conversation=conv.id,
        max_output_tokens=4,
        temperature=0.0,
    )
    assert res.conversation.id == conv.id

    items = client.conversations.items.list(conv.id, order="asc")
    assert len(items.data) >= 2
    assert any(item.role == "assistant" for item in items.data)
