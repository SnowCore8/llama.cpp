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


def test_responses_background_stream_reattach():
    global server
    server.n_ctx = 4096
    server.n_slots = 1
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    # occupy the only slot, the second response stays in progress while we attach to it
    busy = client.responses.create(
        model="gpt-4.1",
        input="Book " * 20,
        max_output_tokens=1024,
        temperature=0.0,
        background=True,
    )
    assert busy.status == "in_progress"

    res = client.responses.create(
        model="gpt-4.1",
        input="Book",
        max_output_tokens=32,
        temperature=0.0,
        background=True,
    )
    assert res.status == "in_progress"

    events = []
    with client.responses.retrieve(res.id, stream=True) as stream:
        for event in stream:
            events.append(event)
            if len(events) == 2:
                # the generation has not started yet, so this is a live subscription
                assert client.responses.retrieve(res.id).status == "in_progress"

    seqs = [event.sequence_number for event in events]
    types = [event.type for event in events]
    assert types[0] == "response.created"
    assert seqs == list(range(len(seqs)))
    assert "response.output_text.delta" in types
    assert types[-1] in ("response.completed", "response.incomplete")
    delta_text = "".join(event.delta for event in events if event.type == "response.output_text.delta")
    assert delta_text == events[-1].response.output_text

    # the log is kept after completion, a reconnect replays the same events
    replay = list(client.responses.retrieve(res.id, stream=True))
    assert [event.sequence_number for event in replay] == seqs
    assert [event.type for event in replay] == types

    # resuming after a sequence number never repeats an event
    mid = seqs[len(seqs) // 2]
    tail = list(client.responses.retrieve(res.id, stream=True, starting_after=mid))
    assert [event.sequence_number for event in tail] == list(range(mid + 1, len(seqs)))
    assert [event.type for event in tail] == types[mid + 1:]


def test_responses_annotations_and_file_id():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(model="gpt-4.1", input="Book", max_output_tokens=4, temperature=0.0)
    assert res.output[0].content[0].annotations == []

    try:
        client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": [{"type": "input_file", "file_id": "file-123"}]}],
        )
        assert False
    except BadRequestError as e:
        assert e.status_code == 400
        assert "file not found" in e.body["message"]

    uploaded = client.files.create(file=("notes.txt", b"Book"), purpose="assistants")
    assert uploaded.object == "file"
    try:
        res = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": [
                {"type": "input_file", "file_id": uploaded.id},
                {"type": "input_text", "text": "Say hi"},
            ]}],
            max_output_tokens=4,
            temperature=0.0,
        )
        assert res.output_text
    finally:
        client.files.delete(uploaded.id)


def test_responses_background_stream_subscribe():
    global server
    server.n_ctx = 2048
    server.n_slots = 1
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    events = []
    with client.responses.create(
        model="gpt-4.1",
        input="Book " * 20,
        max_output_tokens=512,
        temperature=0.0,
        background=True,
        stream=True,
    ) as stream:
        for event in stream:
            events.append(event)
            if len(events) == 2:
                res_id = events[0].response.id
                # the POST connection is only the first subscriber: the response is stored and
                # another connection can attach while it is still running
                assert client.responses.retrieve(res_id).status == "in_progress"

    seqs = [event.sequence_number for event in events]
    types = [event.type for event in events]
    assert types[0] == "response.created"
    assert seqs == list(range(len(seqs)))
    assert "response.output_text.delta" in types
    assert types[-1] in ("response.completed", "response.incomplete")

    # a GET subscriber reads the same log, from the start or from a sequence number
    replay = list(client.responses.retrieve(res_id, stream=True))
    assert [event.sequence_number for event in replay] == seqs
    assert [event.type for event in replay] == types

    mid = seqs[len(seqs) // 2]
    tail = list(client.responses.retrieve(res_id, stream=True, starting_after=mid))
    assert [event.sequence_number for event in tail] == list(range(mid + 1, len(seqs)))


def test_responses_cancel_stream_consistent():
    global server
    server.n_ctx = 2048
    server.n_slots = 1
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book " * 20,
        max_output_tokens=512,
        temperature=0.0,
        background=True,
    )
    assert res.status == "in_progress"

    events = []
    with client.responses.retrieve(res.id, stream=True) as stream:
        for event in stream:
            events.append(event)
            if len(events) == 3:
                cancelled = client.responses.cancel(res.id)
                assert cancelled.status == "cancelled"
                # the store is already cancelled before the subscriber gets the terminal event
                assert client.responses.retrieve(res.id).status == "cancelled"

    assert events[-1].type == "response.completed"
    assert events[-1].response.status == "cancelled"
    assert [event.sequence_number for event in events] == list(range(len(events)))


def test_responses_delete_in_progress():
    global server
    server.n_ctx = 2048
    server.n_slots = 1
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book " * 20,
        max_output_tokens=512,
        temperature=0.0,
        background=True,
    )
    assert res.status == "in_progress"

    events = []
    with client.responses.retrieve(res.id, stream=True) as stream:
        for event in stream:
            events.append(event)
            if len(events) == 3:
                assert client.responses.delete(res.id) is None
                with pytest.raises(NotFoundError):
                    client.responses.retrieve(res.id)

    # the delete cancels the generation, the subscriber still sees the terminal event
    assert events[-1].type == "response.completed"
    assert events[-1].response.status == "cancelled"
    # and the cancelled response is not written back into the store
    time.sleep(1.5)
    with pytest.raises(NotFoundError):
        client.responses.retrieve(res.id)


def test_responses_log_ttl_sweeper(monkeypatch):
    global server
    # a two second replay window lets the test observe the sweeper without waiting 5 minutes
    monkeypatch.setenv("LLAMA_SERVER_RESPONSES_TTL", "2")
    server.n_ctx = 512
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.responses.create(
        model="gpt-4.1",
        input="Book",
        max_output_tokens=8,
        temperature=0.0,
        background=True,
    )

    got = res
    for _ in range(100):
        got = client.responses.retrieve(res.id)
        if got.status in ("completed", "incomplete"):
            break
        time.sleep(0.02)
    assert got.status in ("completed", "incomplete")

    # right after completion the recorded events are still served
    recorded = list(client.responses.retrieve(res.id, stream=True))
    assert any(event.type == "response.output_text.delta" for event in recorded)

    # the sweeper drops the log, then GET falls back to the synthesized replay without deltas
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.5)
        replay = list(client.responses.retrieve(res.id, stream=True))
        if not any(event.type == "response.output_text.delta" for event in replay):
            break
    else:
        raise AssertionError("the event log was not evicted by the TTL sweeper")
    assert replay[0].type == "response.created"
    assert replay[-1].type in ("response.completed", "response.incomplete")
    # only the replay log is dropped, the stored response itself stays readable
    assert client.responses.retrieve(res.id).status in ("completed", "incomplete")
