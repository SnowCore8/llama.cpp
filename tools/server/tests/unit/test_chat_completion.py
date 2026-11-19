import pytest
from openai import OpenAI, BadRequestError, NotFoundError
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


@pytest.mark.parametrize(
    "model,system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason,jinja,chat_template",
    [
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", False, None),
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", True, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", False, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True,  None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True, 'chatml'),
        (None, "Book", "What is the best book", 8, "^ blue",                    23, 8, "length", True, "This is not a chat template, it is"),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length", False, None),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length", True, None),
        (None, "Book", [{"type": "text", "text": "What is"}, {"type": "text", "text": "the best book"}], 8, "Whillicter", 79, 8, "length", False, None),
        (None, "Book", [{"type": "text", "text": "What is"}, {"type": "text", "text": "the best book"}], 8, "Whillicter", 79, 8, "length", True, None),
    ]
)
def test_chat_completion(model, system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason, jinja, chat_template):
    global server
    server.jinja = jinja
    server.chat_template = chat_template
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    })
    assert res.status_code == 200
    assert "cmpl" in res.body["id"] # make sure the completion id has the expected format
    assert res.body["system_fingerprint"].startswith("b")
    # we no longer reflect back the model name, see https://github.com/ggml-org/llama.cpp/pull/17668
    # assert res.body["model"] == model if model is not None else server.model_alias
    assert res.body["usage"]["prompt_tokens"] == n_prompt
    assert res.body["usage"]["completion_tokens"] == n_predicted
    details = res.body["usage"]["completion_tokens_details"]
    assert details["reasoning_tokens"] >= 0
    assert details["text_tokens"] >= 0
    assert details["reasoning_tokens"] + details["text_tokens"] == res.body["usage"]["completion_tokens"]
    choice = res.body["choices"][0]
    assert "assistant" == choice["message"]["role"]
    assert match_regex(re_content, choice["message"]["content"]), f'Expected {re_content}, got {choice["message"]["content"]}'
    assert choice["finish_reason"] == finish_reason


def test_chat_completion_cached_tokens():
    global server
    server.n_slots = 1
    server.start()
    seq = [
        ("1 2 3 4 5 6", 77, 0),
        ("1 2 3 4 5 6", 77, 76),
        ("1 2 3 4 5 9", 77, 51),
        ("1 2 3 9 9 9", 77, 47),
    ]
    for user_prompt, n_prompt, n_cache in seq:
        res = server.make_request("POST", "/chat/completions", data={
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": "Test"},
                {"role": "user", "content": user_prompt},
            ],
        })
        assert res.body["usage"]["prompt_tokens"] == n_prompt
        assert res.body["usage"]["prompt_tokens_details"]["cached_tokens"] == n_cache

@pytest.mark.parametrize(
    "system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason",
    [
        ("Book", "What is the best book", 8, "(Suddenly)+", 77, 8, "length"),
        ("You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length"),
    ]
)
def test_chat_completion_stream(system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason):
    global server
    server.model_alias = "llama-test-model"
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    })
    content = ""
    last_cmpl_id = None
    for i, data in enumerate(res):
        if data["choices"]:
            choice = data["choices"][0]
            if i == 0:
                # Check first role message for stream=True
                assert choice["delta"]["content"] is None
                assert choice["delta"]["role"] == "assistant"
            else:
                assert "role" not in choice["delta"]
            assert data["system_fingerprint"].startswith("b")
            assert data["model"] == "llama-test-model"
            if last_cmpl_id is None:
                last_cmpl_id = data["id"]
            assert last_cmpl_id == data["id"] # make sure the completion id is the same for all events in the stream
            if choice["finish_reason"] in ["stop", "length"]:
                assert "content" not in choice["delta"]
                assert match_regex(re_content, content)
                assert choice["finish_reason"] == finish_reason
            else:
                assert choice["finish_reason"] is None
                content += choice["delta"]["content"] or ''
        else:
            assert data["usage"]["prompt_tokens"] == n_prompt
            assert data["usage"]["completion_tokens"] == n_predicted


def test_chat_completion_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=8,
        seed=42,
        temperature=0.8,
    )
    assert res.system_fingerprint is not None and res.system_fingerprint.startswith("b")
    assert res.choices[0].finish_reason == "length"
    assert res.choices[0].message.content is not None
    assert match_regex("(Suddenly)+", res.choices[0].message.content)


def test_chat_template():
    global server
    server.chat_template = "llama3"
    server.debug = True  # to get the "__verbose" object in the response
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"] == "<s> <|start_header_id|>system<|end_header_id|>\n\nBook<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("prefill,re_prefill", [
    ("Whill", "Whill"),
    ([{"type": "text", "text": "Wh"}, {"type": "text", "text": "ill"}], "Wh\n\nill"),
])
def test_chat_template_assistant_prefill(prefill, re_prefill):
    global server
    server.jinja = True
    server.chat_template_file = "../../../models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja"
    server.debug = True  # to get the "__verbose" object in the response
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
            {"role": "assistant", "content": prefill},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"].endswith(f"<|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{re_prefill}")


def test_chat_template_continue_final_message_vllm_compat():
    """continue_final_message is the vLLM/transformers explicit alias for the prefill_assistant heuristic.
    Both must produce the same prompt."""
    global server
    server.jinja = True
    server.chat_template_file = "../../../models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja"
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
            {"role": "assistant", "content": "Whill"},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"].endswith("<|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nWhill")


def test_chat_template_continue_final_message_mutual_exclusion():
    """add_generation_prompt and continue_final_message both set to true must be rejected"""
    global server
    server.chat_template = "llama3"
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "add_generation_prompt": True,
        "continue_final_message": True,
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    })
    assert res.status_code == 400


def test_apply_chat_template():
    global server
    server.chat_template = "command-r"
    server.start()
    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content":"Hi there"},
        ]
    })
    assert res.status_code == 200
    assert "prompt" in res.body
    assert res.body["prompt"] == "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a test.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"


@pytest.mark.parametrize("response_format,n_predicted,re_content", [
    ({"type": "json_object", "schema": {"const": "42"}}, 6, "\"42\""),
    ({"type": "json_object", "schema": {"items": [{"type": "integer"}]}}, 10, "[ -3000 ]"),
    ({"type": "json_schema", "json_schema": {"schema": {"const": "foooooo"}}}, 10, "\"foooooo\""),
    ({"type": "json_schema", "schema": {"const": "foooooo"}}, 10, "\"foooooo\""),
    # the json_schema wrapper takes precedence over a top-level schema, even when its schema is {}
    ({"type": "json_schema", "json_schema": {"schema": {"const": "foooooo"}}, "schema": {"const": "42"}}, 10, "\"foooooo\""),
    ({"type": "json_schema", "json_schema": {"schema": {}}, "schema": {"type": "hiccup"}}, 10, ".+"),
    ({"type": "json_object"}, 10, "(\\{|John)+"),
    ({"type": "sound"}, 0, None),
    # invalid response format (expected to fail)
    ({"type": "json_object", "schema": 123}, 0, None),
    ({"type": "json_object", "schema": {"type": 123}}, 0, None),
    ({"type": "json_object", "schema": {"type": "hiccup"}}, 0, None),
])
def test_completion_with_response_format(response_format: dict, n_predicted: int, re_content: str | None):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "response_format": response_format,
    })
    if re_content is not None:
        assert res.status_code == 200
        choice = res.body["choices"][0]
        assert match_regex(re_content, choice["message"]["content"])
    else:
        assert res.status_code == 400
        assert "error" in res.body


@pytest.mark.parametrize("jinja,json_schema,n_predicted,re_content", [
    (False, {"const": "42"}, 6, "\"42\""),
    (True, {"const": "42"}, 6, "\"42\""),
])
def test_completion_with_json_schema(jinja: bool, json_schema: dict, n_predicted: int, re_content: str):
    global server
    server.jinja = jinja
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "json_schema": json_schema,
    })
    assert res.status_code == 200, f'Expected 200, got {res.status_code}'
    choice = res.body["choices"][0]
    assert match_regex(re_content, choice["message"]["content"]), f'Expected {re_content}, got {choice["message"]["content"]}'


@pytest.mark.parametrize("jinja,grammar,n_predicted,re_content", [
    (False, 'root ::= "a"{5,5}', 6, "a{5,5}"),
    (True, 'root ::= "a"{5,5}', 6, "a{5,5}"),
])
def test_completion_with_grammar(jinja: bool, grammar: str, n_predicted: int, re_content: str):
    global server
    server.jinja = jinja
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "user", "content": "Does not matter what I say, does it?"},
        ],
        "grammar": grammar,
    })
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert match_regex(re_content, choice["message"]["content"]), choice["message"]["content"]


def test_completion_with_invalid_grammar():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "Does not matter what I say, does it?"},
        ],
        "grammar": "root ::= this is (not valid GBNF",
    })
    assert res.status_code == 400, res.body
    assert "error" in res.body


@pytest.mark.parametrize("messages", [
    None,
    "string",
    [123],
    [{}],
    [{"role": 123}],
    [{"role": "system", "content": 123}],
    # [{"content": "hello"}], # TODO: should not be a valid case
    [{"role": "system", "content": "test"}, {}],
    [{"role": "user", "content": "test"}, {"role": "assistant", "content": "test"}, {"role": "assistant", "content": "test"}],
])
def test_invalid_chat_completion_req(messages):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages,
    })
    assert res.status_code == 400 or res.status_code == 500
    assert "error" in res.body


def test_chat_completion_with_timings_per_token():
    global server
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "test"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    })
    stats_received = False
    for i, data in enumerate(res):
        if i == 0:
            # Check first role message for stream=True
            assert data["choices"][0]["delta"]["content"] is None
            assert data["choices"][0]["delta"]["role"] == "assistant"
            assert "timings" not in data, f'First event should not have timings: {data}'
        else:
            if data["choices"]:
                assert "role" not in data["choices"][0]["delta"]
            else:
                assert "timings" in data
                assert "prompt_per_second" in data["timings"]
                assert "predicted_per_second" in data["timings"]
                assert "predicted_n" in data["timings"]
                assert data["timings"]["predicted_n"] <= 10
                stats_received = True
    assert stats_received


def test_logprobs():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
    )
    output_text = res.choices[0].message.content
    aggregated_text = ''
    assert res.choices[0].logprobs is not None
    assert res.choices[0].logprobs.content is not None
    for token in res.choices[0].logprobs.content:
        aggregated_text += token.token
        assert token.logprob <= 0.0
        assert token.bytes is not None
        assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text


def test_logprobs_stream():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
        stream=True,
    )
    output_text = ''
    aggregated_text = ''
    for i, data in enumerate(res):
        if data.choices:
            choice = data.choices[0]
            if i == 0:
                # Check first role message for stream=True
                assert choice.delta.content is None
                assert choice.delta.role == "assistant"
            else:
                assert choice.delta.role is None
                if choice.finish_reason is None:
                    if choice.delta.content:
                        output_text += choice.delta.content
                    assert choice.logprobs is not None
                    assert choice.logprobs.content is not None
                    for token in choice.logprobs.content:
                        aggregated_text += token.token
                        assert token.logprob <= 0.0
                        assert token.bytes is not None
                        assert token.top_logprobs is not None
                        assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text


def test_logit_bias():
    global server
    server.start()

    exclude = ["i", "I", "the", "The", "to", "a", "an", "be", "is", "was", "but", "But", "and", "And", "so", "So", "you", "You", "he", "He", "she", "She", "we", "We", "they", "They", "it", "It", "his", "His", "her", "Her", "book", "Book"]

    res = server.make_request("POST", "/tokenize", data={
        "content": " " + " ".join(exclude) + " ",
    })
    assert res.status_code == 200
    tokens = res.body["tokens"]
    logit_bias = {tok: -100 for tok in tokens}

    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=64,
        logit_bias=logit_bias
    )
    output_text = res.choices[0].message.content
    assert output_text
    assert all(output_text.find(" " + tok + " ") == -1 for tok in exclude)

def test_context_size_exceeded():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ] * 100, # make the prompt too long
    })
    assert res.status_code == 400
    assert "error" in res.body
    assert res.body["error"]["type"] == "invalid_request_error"
    assert res.body["error"]["code"] == "context_length_exceeded"
    assert res.body["error"]["n_prompt_tokens"] > 0
    assert server.n_ctx is not None
    assert server.n_slots is not None
    assert res.body["error"]["n_ctx"] == server.n_ctx // server.n_slots


def test_context_size_exceeded_stream():
    global server
    server.start()
    try:
        for _ in server.make_stream_request("POST", "/chat/completions", data={
            "messages": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ] * 100, # make the prompt too long
            "stream": True}):
                pass
        assert False, "Should have failed"
    except ServerError as e:
        assert e.code == 400
        assert "error" in e.body
        assert e.body["error"]["type"] == "invalid_request_error"
        assert e.body["error"]["code"] == "context_length_exceeded"
        assert e.body["error"]["n_prompt_tokens"] > 0
        assert server.n_ctx is not None
        assert server.n_slots is not None
        assert e.body["error"]["n_ctx"] == server.n_ctx // server.n_slots


@pytest.mark.parametrize(
    "n_batch,batch_count,reuse_cache",
    [
        (64, 4, False),
        (64, 2, True),
    ]
)
def test_return_progress(n_batch, batch_count, reuse_cache):
    global server
    server.n_batch = n_batch
    server.n_ctx = 256
    server.n_slots = 1
    server.start()
    def make_cmpl_request():
        return server.make_stream_request("POST", "/chat/completions", data={
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "This is a test" * 10},
            ],
            "stream": True,
            "return_progress": True,
        })
    if reuse_cache:
        # make a first request to populate the cache
        res0 = make_cmpl_request()
        for _ in res0:
            pass # discard the output

    res = make_cmpl_request()
    last_progress = None
    total_batch_count = 0

    for data in res:
        cur_progress = data.get("prompt_progress", None)
        if cur_progress is None:
            continue
        if total_batch_count == 0:
            # first progress report must have n_cache == n_processed
            assert cur_progress["total"] > 0
            assert cur_progress["cache"] == cur_progress["processed"]
            if reuse_cache:
                # when reusing cache, we expect some cached tokens
                assert cur_progress["cache"] > 0
        if last_progress is not None:
            assert cur_progress["total"] == last_progress["total"]
            assert cur_progress["cache"] == last_progress["cache"]
            assert cur_progress["processed"] > last_progress["processed"]
        total_batch_count += 1
        last_progress = cur_progress

    # last progress should indicate completion (all tokens processed)
    assert last_progress is not None
    assert last_progress["total"] > 0
    assert last_progress["processed"] == last_progress["total"]
    assert total_batch_count == batch_count


def test_chat_completions_multiple_choices():
    global server
    server.start()
    # make sure cache can be reused across multiple choices and multiple requests
    # ref: https://github.com/ggml-org/llama.cpp/pull/18663
    for _ in range(2):
        res = server.make_request("POST", "/chat/completions", data={
            "max_tokens": 8,
            "n": 2,
            "messages": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ],
            # test forcing the same slot to be used
            # the scheduler should not be locked up in this case
            "id_slot": 0,
        })
        assert res.status_code == 200
        assert len(res.body["choices"]) == 2
        for choice in res.body["choices"]:
            assert "assistant" == choice["message"]["role"]
            assert choice["finish_reason"] == "length"
        # each choice generates max_tokens tokens, usage counts them all
        # ref: https://github.com/ggml-org/llama.cpp/issues/29451
        assert res.body["usage"]["completion_tokens"] == 16
        assert res.body["usage"]["total_tokens"] == res.body["usage"]["prompt_tokens"] + 16
        # every choice contributes its exact count, summed for the request
        details = res.body["usage"]["completion_tokens_details"]
        assert details["reasoning_tokens"] >= 0
        assert details["reasoning_tokens"] + details["text_tokens"] == res.body["usage"]["completion_tokens"]


def test_chat_completions_token_count():
    global server
    server.start()
    # make sure cache can be reused across multiple choices and multiple requests
    # ref: https://github.com/ggml-org/llama.cpp/pull/18663
    for _ in range(2):
        res = server.make_request("POST", "/chat/completions/input_tokens", data={
            "messages": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ],
        })
        assert res.status_code == 200
        assert res.body["input_tokens"] > 5


def test_verbose_debug():
    global server
    server.start()
    for verbose in [True, False]:
        res = server.make_request("POST", "/chat/completions", data={
            "max_tokens": 2,
            "messages": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ],
            "verbose": verbose,
        })
        assert res.status_code == 200
        if verbose:
            assert "__verbose" in res.body
            assert "Book" in res.body["__verbose"]["prompt"]
        else:
            assert "__verbose" not in res.body


def test_chat_completion_store_crud():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Once upon a time"}],
        max_completion_tokens=8,
        temperature=0.0,
        store=True,
        metadata={"k": "v"},
    )
    assert res.metadata == {"k": "v"}
    assert res.choices[0].message.refusal is None
    assert res.usage.completion_tokens_details is not None

    got = client.chat.completions.retrieve(res.id)
    assert got.id == res.id

    updated = client.chat.completions.update(res.id, metadata={"k": "v2"})
    assert updated.metadata == {"k": "v2"}

    listed = client.chat.completions.list(limit=10)
    assert any(item.id == res.id for item in listed.data)

    messages = client.chat.completions.messages.list(res.id)
    assert len(messages.data) == 1
    assert messages.data[0].role == "assistant"

    deleted = client.chat.completions.delete(res.id)
    assert deleted.deleted is True
    with pytest.raises(NotFoundError):
        client.chat.completions.retrieve(res.id)


def test_chat_completion_store_stream():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    stream = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Once"}],
        max_completion_tokens=4,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        store=True,
    )
    completion_id = None
    usage = None
    for chunk in stream:
        completion_id = chunk.id
        if chunk.usage is not None:
            usage = chunk.usage
    assert completion_id is not None
    assert usage is not None
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens >= 0
    assert usage.completion_tokens_details.text_tokens >= 0
    assert client.chat.completions.retrieve(completion_id).id == completion_id


def test_chat_completion_store_disabled():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Once"}],
        max_completion_tokens=2,
        temperature=0.0,
    )
    with pytest.raises(NotFoundError):
        client.chat.completions.retrieve(res.id)


def test_chat_error_envelope():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    try:
        client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "hi"}],
            temperature=3.0,
        )
        assert False
    except BadRequestError as e:
        assert e.status_code == 400
        assert e.body["type"] == "invalid_request_error"
        assert e.body["param"] == "temperature"
        assert e.body["code"] == "invalid_value"


CUSTOM_TOOL = {
    "type": "custom",
    "custom": {
        "name": "run_code",
        "description": "Run python code",
        "format": {"type": "text"},
    },
}

QWEN35_TEMPLATE = "../../../models/templates/Qwen3.5-4B.jinja"

# Qwen3.5 XML tool call layout, used to drive a deterministic custom tool call
CUSTOM_TOOL_CALL_COMPLETE = (
    "<tool_call>\n<function=run_code>\n<parameter=input>\n"
    "print(1)\n</parameter>\n</function>\n</tool_call>"
)
CUSTOM_TOOL_CALL_OPEN = "<tool_call>\n<function=run_code>\n<parameter=input>\n"


def start_custom_tool_server():
    global server
    server.jinja = True
    server.chat_template_file = QWEN35_TEMPLATE
    server.n_ctx = 2048
    server.n_slots = 1
    server.temperature = 0.0
    server.start()


def test_custom_tools_request_validation():
    start_custom_tool_server()
    base = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }

    def expect_400(payload, needle):
        res = server.make_request("POST", "/chat/completions", data=payload)
        assert res.status_code == 400, res.body
        assert res.body["error"]["type"] == "invalid_request_error"
        assert needle in res.body["error"]["message"], res.body

    expect_400({**base, "tools": [{"type": "custom", "custom": {}}]}, "custom.name")
    expect_400({**base, "tools": [{"type": "custom", "custom": {"name": "x", "format": {"type": "json"}}}]}, "custom.format.type")
    expect_400({**base, "tools": [{"type": "custom", "custom": {"name": "x", "format": {"type": "grammar", "grammar": {"syntax": "ebnf", "definition": "x"}}}}]}, "custom.format.grammar.syntax")
    expect_400({**base, "tools": [{"type": "custom", "custom": {"name": "x", "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": ""}}}}]}, "custom.format.grammar.definition")
    expect_400({**base, "tools": [{"type": "web_search"}]}, "Unsupported tool type")
    expect_400({**base, "tools": [CUSTOM_TOOL], "tool_choice": {"type": "custom", "custom": {"name": "missing"}}}, "unknown tool")

    # valid custom formats are accepted
    for fmt in ({"type": "text"}, {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: /.+/"}}):
        tool = {"type": "custom", "custom": {"name": "run_code", "description": "Run code", "format": fmt}}
        res = server.make_request("POST", "/chat/completions", data={**base, "tools": [tool]})
        assert res.status_code == 200, res.body

    # a named custom tool choice keeps the tool and rewrites the choice to required
    res = server.make_request("POST", "/chat/completions", data={
        **base,
        "tools": [CUSTOM_TOOL],
        "tool_choice": {"type": "custom", "custom": {"name": "run_code"}},
    })
    assert res.status_code == 200, res.body


def test_custom_tool_call_response():
    start_custom_tool_server()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "user", "content": "Run print(1)"},
            {"role": "assistant", "content": CUSTOM_TOOL_CALL_COMPLETE},
        ],
        "continue_final_message": "content",
        "add_generation_prompt": False,
        "tools": [CUSTOM_TOOL],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert res.status_code == 200, res.body
    tool_calls = res.body["choices"][0]["message"].get("tool_calls")
    assert tool_calls is not None and len(tool_calls) == 1, res.body
    tool_call = tool_calls[0]
    assert tool_call["type"] == "custom", tool_call
    assert tool_call["custom"]["name"] == "run_code", tool_call
    assert tool_call["custom"]["input"] == "print(1)", tool_call
    assert tool_call.get("id"), tool_call
    assert "function" not in tool_call, tool_call


def test_custom_tool_call_stream():
    start_custom_tool_server()
    tool_calls = {}
    finish_reason = None
    for chunk in server.make_stream_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "user", "content": "Run print(1)"},
            {"role": "assistant", "content": CUSTOM_TOOL_CALL_OPEN},
        ],
        "continue_final_message": "content",
        "add_generation_prompt": False,
        "tools": [CUSTOM_TOOL],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 8,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }):
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            for tool_call in delta.get("tool_calls") or []:
                assert "function" not in tool_call, tool_call
                index = tool_call.get("index", 0)
                entry = tool_calls.setdefault(index, {"types": set(), "name": None, "id": None, "inputs": []})
                if tool_call.get("type") is not None:
                    entry["types"].add(tool_call["type"])
                custom = tool_call.get("custom") or {}
                if custom.get("name") is not None:
                    entry["name"] = custom["name"]
                if tool_call.get("id") is not None:
                    entry["id"] = tool_call["id"]
                if custom.get("input") is not None:
                    entry["inputs"].append(custom["input"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    assert len(tool_calls) == 1, tool_calls
    (_, entry), = tool_calls.items()
    assert entry["types"] == {"custom"}, entry
    assert entry["name"] == "run_code", entry
    assert entry["id"], entry
    assert entry["inputs"] and all(isinstance(part, str) for part in entry["inputs"]), entry
    assert finish_reason in ("length", "tool_calls"), finish_reason


def test_custom_tool_result_backfill():
    start_custom_tool_server()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "user", "content": "Run print(1)"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "custom", "custom": {"name": "run_code", "input": "print(1)"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "1"},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [CUSTOM_TOOL],
        "max_tokens": 2,
    })
    assert res.status_code == 200, res.body
    assert res.body["choices"][0]["message"]["role"] == "assistant"


def test_web_search_options_ignored():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [{"role": "user", "content": "Book"}],
        "web_search_options": {"search_context_size": "low"},
        "max_tokens": 4,
    })
    assert res.status_code == 200, res.body
