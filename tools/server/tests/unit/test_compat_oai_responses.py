import time

import pytest
from openai import OpenAI, BadRequestError, NotFoundError
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_responses_stream_with_llama_telemetry():
    global server
    server.n_ctx = 256
    server.n_batch = 32
    server.n_slots = 1
    server.start()

    saw_progress = False
    saw_delta_timings = False
    completed = None

    res = server.make_stream_request("POST", "/responses", data={
        "input": "This is a test" * 10,
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    })

    for data in res:
        if "prompt_progress" in data:
            assert data["type"] == "response.in_progress"
            assert data["prompt_progress"]["total"] > 0
            assert data["prompt_progress"]["processed"] >= data["prompt_progress"]["cache"]
            saw_progress = True
        if "timings" in data:
            assert "prompt_per_second" in data["timings"]
            assert "predicted_per_second" in data["timings"]
            if data["type"] == "response.output_text.delta":
                saw_delta_timings = True
        if data["type"] in ("response.completed", "response.incomplete"):
            completed = data

    assert saw_progress
    assert saw_delta_timings
    assert completed is not None
    assert completed["response"]["status"] in ("completed", "incomplete")
    assert "usage" in completed["response"]
    assert "timings" in completed


def test_responses_store_and_previous_response_id():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    first = client.responses.create(model="gpt-4.1", input="Book", max_output_tokens=4, temperature=0.0)
    assert first.id.startswith("resp_")

    fetched = client.responses.retrieve(first.id)
    assert fetched.id == first.id
    assert fetched.status == first.status

    items = client.responses.input_items.list(first.id)
    assert len(items.data) == 1
    assert items.data[0].role == "user"

    second = client.responses.create(
        model="gpt-4.1",
        input="Again",
        previous_response_id=first.id,
        max_output_tokens=4,
        temperature=0.0,
    )
    assert second.previous_response_id == first.id

    assert client.responses.delete(first.id) is None
    with pytest.raises(NotFoundError):
        client.responses.retrieve(first.id)


def test_responses_store_disabled():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(model="gpt-4.1", input="Book", max_output_tokens=4, temperature=0.0, store=False)
    with pytest.raises(NotFoundError):
        client.responses.retrieve(res.id)

    with pytest.raises(Exception):
        client.responses.create(
            model="gpt-4.1",
            input="Again",
            previous_response_id=res.id,
            max_output_tokens=4,
        )


def test_responses_background():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book",
        max_output_tokens=8,
        temperature=0.0,
        background=True,
    )
    assert res.background is True

    got = res
    for _ in range(100):
        got = client.responses.retrieve(res.id)
        if got.status in ("completed", "incomplete", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert got.status in ("completed", "incomplete")


def test_responses_cancel():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book " * 20,
        max_output_tokens=128,
        temperature=0.0,
        background=True,
    )
    cancelled = client.responses.cancel(res.id)
    assert cancelled.status == "cancelled"
    assert client.responses.retrieve(res.id).status == "cancelled"


def test_responses_logprobs():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book",
        max_output_tokens=4,
        temperature=0.0,
        include=["message.output_text.logprobs"],
        top_logprobs=1,
    )
    logprobs = res.output[0].content[0].logprobs
    assert logprobs is not None
    assert len(logprobs) > 0


def test_responses_background_stream():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    stream = client.responses.create(
        model="gpt-4.1",
        input="Book",
        max_output_tokens=4,
        temperature=0.0,
        background=True,
        stream=True,
    )
    types = [event.type for event in stream]
    assert types[0] == "response.created"
    assert types[-1] in ("response.completed", "response.incomplete")


def test_responses_retrieve_stream_replay():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(model="gpt-4.1", input="Book", max_output_tokens=4, temperature=0.0)
    stream = client.responses.retrieve(res.id, stream=True)
    types = [event.type for event in stream]
    assert types[0] == "response.created"
    assert types[-1] in ("response.completed", "response.incomplete")


def test_responses_error_envelope():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    try:
        client.responses.create(model="gpt-4.1", input="hi", temperature=3.0)
        assert False
    except BadRequestError as e:
        assert e.status_code == 400
        assert e.body["type"] == "invalid_request_error"
        assert e.body["param"] == "temperature"
        assert e.body["code"] is None
