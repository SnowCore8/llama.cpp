#!/usr/bin/env python3
import pytest
import base64
import json
import requests

from utils import *

server: ServerProcess


def get_test_image_base64() -> str:
    """Get a test image in base64 format"""
    # Use the same test image as test_vision_api.py
    IMG_URL = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
    response = requests.get(IMG_URL)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2-anthropic"
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048


@pytest.fixture
def vision_server():
    """Separate fixture for vision tests that require multimodal support"""
    global server
    server = ServerPreset.tinygemma3()
    server.offline = False  # Allow downloading the model
    server.model_alias = "tinygemma3-anthropic"
    server.n_slots = 1
    return server


# Basic message tests

def test_anthropic_messages_basic():
    """Test basic Anthropic messages endpoint"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert res.body["type"] == "message", f"Expected type 'message', got {res.body.get('type')}"
    assert res.body["role"] == "assistant", f"Expected role 'assistant', got {res.body.get('role')}"
    assert "content" in res.body, "Missing 'content' field"
    assert isinstance(res.body["content"], list), "Content should be an array"
    assert len(res.body["content"]) > 0, "Content array should not be empty"
    assert res.body["content"][0]["type"] == "text", "First content block should be text"
    assert "text" in res.body["content"][0], "Text content block missing 'text' field"
    assert res.body["stop_reason"] in ["end_turn", "max_tokens"], f"Invalid stop_reason: {res.body.get('stop_reason')}"
    assert "usage" in res.body, "Missing 'usage' field"
    assert "cache_read_input_tokens" in res.body["usage"], "Missing usage.cache_read_input_tokens"
    assert "input_tokens" in res.body["usage"], "Missing usage.input_tokens"
    assert "output_tokens" in res.body["usage"], "Missing usage.output_tokens"
    assert res.body["usage"]["cache_creation_input_tokens"] == 0, "Missing usage.cache_creation_input_tokens"
    assert "thinking_tokens" in res.body["usage"]["output_tokens_details"], "Missing usage.output_tokens_details.thinking_tokens"
    assert isinstance(res.body["usage"]["cache_read_input_tokens"], int), "cache_read_input_tokens should be integer"
    assert isinstance(res.body["usage"]["input_tokens"], int), "input_tokens should be integer"
    assert isinstance(res.body["usage"]["output_tokens"], int), "output_tokens should be integer"
    assert res.body["usage"]["output_tokens"] > 0, "Should have generated some tokens"
    # Anthropic API should NOT include timings
    assert "timings" not in res.body, "Anthropic API should not include timings field"


def test_anthropic_messages_with_system():
    """Test messages with system prompt"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0


def test_anthropic_messages_multipart_content():
    """Test messages with multipart content blocks"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " the answer?"}
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_messages_conversation():
    """Test multi-turn conversation"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Streaming tests

def test_anthropic_messages_streaming():
    """Test streaming messages"""
    server.start()

    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 30,
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "stream": True
    })

    events = []
    for data in res:
        # Each event should have type and other fields
        assert "type" in data, f"Missing 'type' in event: {data}"
        events.append(data)

    # Verify event sequence
    event_types = [e["type"] for e in events]
    assert "message_start" in event_types, "Missing message_start event"
    assert "content_block_start" in event_types, "Missing content_block_start event"
    assert "content_block_delta" in event_types, "Missing content_block_delta event"
    assert "content_block_stop" in event_types, "Missing content_block_stop event"
    assert "message_delta" in event_types, "Missing message_delta event"
    assert "message_stop" in event_types, "Missing message_stop event"

    # Check message_start structure
    message_start = next(e for e in events if e["type"] == "message_start")
    assert "message" in message_start, "message_start missing 'message' field"
    assert message_start["message"]["type"] == "message"
    assert message_start["message"]["role"] == "assistant"
    assert message_start["message"]["content"] == []
    assert "usage" in message_start["message"]
    assert message_start["message"]["usage"]["input_tokens"] > 0

    # Check content_block_start
    block_start = next(e for e in events if e["type"] == "content_block_start")
    assert "index" in block_start, "content_block_start missing 'index'"
    assert block_start["index"] == 0, "First content block should be at index 0"
    assert "content_block" in block_start
    assert block_start["content_block"]["type"] == "text"

    # Check content_block_delta
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    assert len(deltas) > 0, "Should have at least one content_block_delta"
    for delta in deltas:
        assert "index" in delta
        assert "delta" in delta
        assert delta["delta"]["type"] == "text_delta"
        assert "text" in delta["delta"]

    # Check content_block_stop
    block_stop = next(e for e in events if e["type"] == "content_block_stop")
    assert "index" in block_stop
    assert block_stop["index"] == 0

    # Check message_delta
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert "delta" in message_delta
    assert "stop_reason" in message_delta["delta"]
    assert message_delta["delta"]["stop_reason"] in ["end_turn", "max_tokens"]
    assert "usage" in message_delta
    assert message_delta["usage"]["output_tokens"] > 0

    # Check message_stop
    message_stop = next(e for e in events if e["type"] == "message_stop")
    # message_stop should NOT have timings for Anthropic API
    assert "timings" not in message_stop, "Anthropic streaming should not include timings"


# Token counting tests

def test_anthropic_count_tokens():
    """Test token counting endpoint"""
    server.start()

    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello world"}
        ]
    })

    assert res.status_code == 200
    assert "input_tokens" in res.body
    assert isinstance(res.body["input_tokens"], int)
    assert res.body["input_tokens"] > 0
    # Should only have input_tokens, no other fields
    assert "output_tokens" not in res.body


def test_anthropic_count_tokens_with_system():
    """Test token counting with system prompt"""
    server.start()

    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["input_tokens"] > 0


def test_anthropic_count_tokens_no_max_tokens():
    """Test that count_tokens doesn't require max_tokens"""
    server.start()

    # max_tokens is NOT required for count_tokens
    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert "input_tokens" in res.body


# Tool use tests

def test_anthropic_tool_use_basic():
    """Test basic tool use"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 200,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }],
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0

    # Check if model used the tool (it might not always, depending on the model)
    content_types = [block.get("type") for block in res.body["content"]]

    if "tool_use" in content_types:
        # Model used the tool
        assert res.body["stop_reason"] == "tool_use"

        # Find the tool_use block
        tool_block = next(b for b in res.body["content"] if b.get("type") == "tool_use")
        assert "id" in tool_block
        assert "name" in tool_block
        assert tool_block["name"] == "get_weather"
        assert "input" in tool_block
        assert isinstance(tool_block["input"], dict)


def test_anthropic_tool_result():
    """Test sending tool results back

    This test verifies that tool_result blocks are properly converted to
    role="tool" messages internally. Without proper conversion, this would
    fail with a 500 error: "unsupported content[].type" because tool_result
    blocks would remain in the user message content array.
    """
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "test123",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "test123",
                        "content": "The weather is sunny, 25°C"
                    }
                ]
            }
        ]
    })

    # This would be 500 with the old bug where tool_result blocks weren't converted
    assert res.status_code == 200
    assert res.body["type"] == "message"
    # Model should respond to the tool result
    assert len(res.body["content"]) > 0
    assert res.body["content"][0]["type"] == "text"


def test_anthropic_tool_result_with_text():
    """Test tool result mixed with text content

    This tests the edge case where a user message contains both text and
    tool_result blocks. The server must properly split these into separate
    messages: a user message with text, followed by tool messages.
    Without proper handling, this would fail with 500: "unsupported content[].type"
    """
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "Sunny, 25°C"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0


def test_anthropic_tool_result_with_image():
    """Test tool result containing mixed text and image blocks

    Verifies that image blocks inside Anthropic tool_result content are
    properly converted to OpenAI image_url format rather than being
    silently dropped. With a non-multimodal model, the converted image
    triggers a clear error message instead of being ignored.
    """
    server.jinja = True
    server.start()

    # Small 1x1 red PNG image in base64 (same as vision tests)
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What is in this image?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "read",
                        "input": {"file": "test.png"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": [
                            {"type": "text", "text": "File: test.png"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": red_pixel_png
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    })

    # Without the fix, image block would cause "unsupported content[].type"
    # With the fix, image is converted to image_url but tinyllama doesn't support images
    assert res.status_code == 500
    assert "image input is not supported" in res.body.get("error", {}).get("message", "").lower()


def test_anthropic_tool_result_error():
    """Test tool result with error flag"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "test123",
                        "name": "get_weather",
                        "input": {"location": "InvalidCity"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "test123",
                        "is_error": True,
                        "content": "City not found"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_tool_streaming():
    """Test streaming with tool use"""
    server.jinja = True
    server.start()

    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 200,
        "stream": True,
        "tools": [{
            "name": "calculator",
            "description": "Calculate math",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }],
        "messages": [
            {"role": "user", "content": "Calculate 2+2"}
        ]
    })

    events = []
    for data in res:
        events.append(data)

    event_types = [e["type"] for e in events]

    # Should have basic events
    assert "message_start" in event_types
    assert "message_stop" in event_types

    # If tool was used, check for proper tool streaming
    if any(e.get("type") == "content_block_start" and
           e.get("content_block", {}).get("type") == "tool_use"
           for e in events):
        # Find tool use block start
        tool_starts = [e for e in events if
                      e.get("type") == "content_block_start" and
                      e.get("content_block", {}).get("type") == "tool_use"]

        assert len(tool_starts) > 0, "Should have tool_use content_block_start"

        # Check index is correct (should be 0 if no text, 1 if there's text)
        tool_start = tool_starts[0]
        assert "index" in tool_start
        assert tool_start["content_block"]["type"] == "tool_use"
        assert "name" in tool_start["content_block"]


# Vision/multimodal tests

def test_anthropic_vision_format_accepted():
    """Test that Anthropic vision format is accepted (format validation only)"""
    server.start()

    # Small 1x1 red PNG image in base64
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": red_pixel_png
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is this?"
                    }
                ]
            }
        ]
    })

    # Server accepts the format but tinyllama doesn't support images
    # So it should return 500 with clear error message about missing mmproj
    assert res.status_code == 500
    assert "image input is not supported" in res.body.get("error", {}).get("message", "").lower()


def test_anthropic_vision_base64_with_multimodal_model(vision_server):
    """Test vision with base64 image using Anthropic format with multimodal model"""
    global server
    server = vision_server
    server.start()

    # Get test image in base64 format
    image_base64 = get_test_image_base64()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is this:\n"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0
    assert res.body["content"][0]["type"] == "text"
    # The model should generate some response about the image
    assert len(res.body["content"][0]["text"]) > 0


# Parameter tests

def test_anthropic_stop_sequences():
    """Test stop_sequences parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "stop_sequences": ["\n", "END"],
        "messages": [
            {"role": "user", "content": "Count to 10"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_temperature():
    """Test temperature parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_top_p():
    """Test top_p parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "top_p": 0.9,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_top_k():
    """Test top_k parameter (llama.cpp specific)"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "top_k": 40,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Error handling tests

def test_anthropic_missing_messages():
    """Test error when messages are missing"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50
        # missing "messages" field
    })

    # Should return an error (400 or 500)
    assert res.status_code >= 400


def test_anthropic_empty_messages():
    """Test permissive handling of empty messages array"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": []
    })

    # Server is permissive and accepts empty messages (provides defaults)
    # This matches the permissive validation design choice
    assert res.status_code == 200
    assert res.body["type"] == "message"


# Content block index tests

def test_anthropic_streaming_content_block_indices():
    """Test that content block indices are correct in streaming"""
    server.jinja = True
    server.start()

    # Request that might produce both text and tool use
    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 400,
        "stream": True,
        "tools": [{
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        }],
        "messages": [
            {"role": "user", "content": "Use the test tool"}
        ]
    })

    events = []
    for data in res:
        events.append(data)

    # Check content_block_start events have sequential indices
    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    if len(block_starts) > 1:
        # If there are multiple blocks, indices should be sequential
        indices = [e["index"] for e in block_starts]
        expected_indices = list(range(len(block_starts)))
        assert indices == expected_indices, f"Expected indices {expected_indices}, got {indices}"

    # Check content_block_stop events match the starts
    block_stops = [e for e in events if e.get("type") == "content_block_stop"]
    start_indices = set(e["index"] for e in block_starts)
    stop_indices = set(e["index"] for e in block_stops)
    assert start_indices == stop_indices, "content_block_stop indices should match content_block_start indices"


# Extended features tests

def test_anthropic_thinking():
    """Test extended thinking parameter"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1025,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024
        },
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_thinking_bounds():
    """Official bounds: budget_tokens >= 1024 and < max_tokens"""
    server.start()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1025,
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "messages": messages,
    })
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "at least 1024" in res.body["error"]["message"]

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": messages,
    })
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "less than 'max_tokens'" in res.body["error"]["message"]

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1025,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": messages,
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"


def test_anthropic_thinking_adaptive_and_display():
    """thinking adaptive carries no budget; thinking.display is validated"""
    server.start()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 64,
        "thinking": {"type": "adaptive"},
        "messages": messages,
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1025,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "verbose"},
        "messages": messages,
    })
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "display" in res.body["error"]["message"]

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 1025,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
        "messages": messages,
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    # the block is kept when the model reasons; its text stays hidden
    for block in res.body["content"]:
        if block.get("type") == "thinking":
            assert block["thinking"] == "", f"display=omitted leaked thinking text: {block['thinking']!r}"


def test_anthropic_error_envelope():
    """errors use the official Anthropic envelope: {"type":"error","error":{...},"request_id"}"""
    server.start()

    # max_tokens is required by the official API
    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}],
    })

    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert res.body["type"] == "error", f"Expected the Anthropic error envelope, got {res.body}"
    assert res.body["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in res.body["error"]["message"]
    assert isinstance(res.body["error"]["message"], str) and res.body["error"]["message"]
    assert "request_id" in res.body


def test_anthropic_max_tokens_required():
    """max_tokens must be a non-negative integer"""
    server.start()
    messages = [{"role": "user", "content": "Hello"}]

    for value in (None, -1, "16", 1.5):
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": value,
            "messages": messages,
        })
        assert res.status_code == 400, f"Expected 400 for max_tokens={value!r}, got {res.status_code}: {res.body}"
        assert "max_tokens" in res.body["error"]["message"]


def test_anthropic_response_fields():
    """official response fields: container / stop_details / service_tier and the usage detail fields"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    assert res.body["container"] is None
    assert res.body["stop_details"] is None

    usage = res.body["usage"]
    assert usage["cache_creation"] == {
        "ephemeral_1h_input_tokens": 0,
        "ephemeral_5m_input_tokens": 0,
    }
    assert usage["inference_geo"] is None
    assert usage["service_tier"] == "standard"
    assert usage["server_tool_use"] == {
        "web_fetch_requests": 0,
        "web_search_requests": 0,
    }


def test_anthropic_cloud_shaped_fields():
    """service_tier / container / inference_geo are validated, then dropped (no local semantics)"""
    server.start()
    messages = [{"role": "user", "content": "Hello"}]

    for field, value in [("service_tier", "turbo"), ("inference_geo", "eu"), ("container", 5)]:
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 16,
            field: value,
            "messages": messages,
        })
        assert res.status_code == 400, f"Expected 400 for {field}={value!r}, got {res.status_code}: {res.body}"

    for field, value in [
        ("service_tier", "standard_only"),
        ("inference_geo", "us"),
        ("container", "cnt_1"),
        ("container", {"id": "cnt_1"}),
    ]:
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 16,
            field: value,
            "messages": messages,
        })
        assert res.status_code == 200, f"Expected 200 for {field}={value!r}, got {res.status_code}: {res.body}"


def test_anthropic_stream_message_delta_usage():
    """message_delta.usage carries the cumulative totals"""
    server.start()

    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    })
    events = list(res)

    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert deltas, f"Expected a message_delta event, got {[e.get('type') for e in events]}"
    usage = deltas[-1]["usage"]
    assert usage["output_tokens"] > 0
    assert usage["input_tokens"] > 0
    assert "cache_read_input_tokens" in usage
    assert "cache_creation_input_tokens" in usage


def test_anthropic_metadata():
    """Test metadata parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "metadata": {
            "user_id": "test_user_123"
        },
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Compatibility tests

def test_anthropic_vs_openai_different_response_format():
    """Verify Anthropic format is different from OpenAI format"""
    server.start()

    # Make OpenAI request
    openai_res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias,
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    # Make Anthropic request
    anthropic_res = server.make_request("POST", "/v1/messages", data={
        "model": server.model_alias,
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert openai_res.status_code == 200
    assert anthropic_res.status_code == 200

    # OpenAI has "object", Anthropic has "type"
    assert "object" in openai_res.body
    assert "type" in anthropic_res.body
    assert openai_res.body["object"] == "chat.completion"
    assert anthropic_res.body["type"] == "message"

    # OpenAI has "choices", Anthropic has "content"
    assert "choices" in openai_res.body
    assert "content" in anthropic_res.body

    # Different usage field names
    assert "prompt_tokens" in openai_res.body["usage"]
    assert "input_tokens" in anthropic_res.body["usage"]
    assert "completion_tokens" in openai_res.body["usage"]
    assert "output_tokens" in anthropic_res.body["usage"]


# Extended thinking tests with reasoning models

# The next two tests cover the input path (conversation history):
# Client sends thinking blocks -> convert_anthropic_to_oai -> reasoning_content -> template

def test_anthropic_thinking_history_in_count_tokens():
    """Test that interleaved thinking blocks in conversation history are not dropped during conversion."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    tool = {
        "name": "list_files",
        "description": "List files",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }

    messages_without_thinking = [
        {"role": "user", "content": "Fix the bug"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "list_files", "input": {"path": "."}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "main.py"}
            ]
        },
    ]

    messages_with_thinking = [
        {"role": "user", "content": "Fix the bug"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should check the project structure first to understand the codebase layout."},
                {"type": "tool_use", "id": "call_1", "name": "list_files", "input": {"path": "."}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "main.py"}
            ]
        },
    ]

    res_without = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": messages_without_thinking,
        "tools": [tool],
    })
    assert res_without.status_code == 200, f"Expected 200: {res_without.body}"

    res_with = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": messages_with_thinking,
        "tools": [tool],
    })
    assert res_with.status_code == 200, f"Expected 200: {res_with.body}"

    # Thinking blocks should increase the token count
    assert res_with.body["input_tokens"] > res_without.body["input_tokens"], \
        f"Expected more tokens with thinking ({res_with.body['input_tokens']}) than without ({res_without.body['input_tokens']})"


def test_anthropic_thinking_history_in_template():
    """Test that reasoning_content from converted interleaved thinking blocks renders in the prompt."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    reasoning_1 = "I should check the project structure first."
    reasoning_2 = "Now I need to read the main file."

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "Fix the bug in main.py"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_1,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "main.py\nutils.py"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_2,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\": \"main.py\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "print('hello')"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        }, {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        }],
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    prompt = res.body["prompt"]

    # Both reasoning_content values should be rendered in <think> tags
    assert reasoning_1 in prompt, f"Expected first reasoning text in prompt: {prompt}"
    assert reasoning_2 in prompt, f"Expected second reasoning text in prompt: {prompt}"
    assert prompt.count("<think>") >= 2, f"Expected at least 2 <think> blocks in prompt: {prompt}"


@pytest.mark.slow
@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_thinking_with_reasoning_model(stream):
    """Test that thinking content blocks are properly returned for reasoning models"""
    global server
    server = ServerProcess()
    server.model_hf_repo = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
    server.model_hf_file = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    server.reasoning_format = "deepseek"
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = 1024
    server.start(timeout_seconds=600)  # large model needs time to download

    if stream:
        res = server.make_stream_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 3072,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            },
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "stream": True
        })

        events = list(res)

        # should have thinking content block events
        thinking_starts = [e for e in events if
            e.get("type") == "content_block_start" and
            e.get("content_block", {}).get("type") == "thinking"]
        assert len(thinking_starts) > 0, "Should have thinking content_block_start event"
        assert thinking_starts[0]["index"] == 0, "Thinking block should be at index 0"
        assert "signature" in thinking_starts[0]["content_block"], \
            "Thinking content_block_start should carry a signature field"

        # should have thinking_delta events
        thinking_deltas = [e for e in events if
            e.get("type") == "content_block_delta" and
            e.get("delta", {}).get("type") == "thinking_delta"]
        assert len(thinking_deltas) > 0, "Should have thinking_delta events"

        # should have signature_delta event before thinking block closes (Anthropic API requirement)
        signature_deltas = [e for e in events if
            e.get("type") == "content_block_delta" and
            e.get("delta", {}).get("type") == "signature_delta"]
        assert len(signature_deltas) > 0, "Should have signature_delta event for thinking block"

        # should have text block after thinking
        text_starts = [e for e in events if
            e.get("type") == "content_block_start" and
            e.get("content_block", {}).get("type") == "text"]
        assert len(text_starts) > 0, "Should have text content_block_start event"
        assert text_starts[0]["index"] == 1, "Text block should be at index 1 (after thinking)"
    else:
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 3072,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            },
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        })

        assert res.status_code == 200
        assert res.body["type"] == "message"

        content = res.body["content"]
        assert len(content) >= 2, "Should have at least thinking and text blocks"

        # first block should be thinking
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        assert len(thinking_blocks) > 0, "Should have thinking content block"
        assert "thinking" in thinking_blocks[0], "Thinking block should have 'thinking' field"
        assert len(thinking_blocks[0]["thinking"]) > 0, "Thinking content should not be empty"
        assert "signature" in thinking_blocks[0], "Thinking block should have 'signature' field (Anthropic API requirement)"

        # should also have text block
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(text_blocks) > 0, "Should have text content block"


# Anthropic request deepening: output_config, tool_choice shapes, cache_control.

def test_anthropic_tool_choice_none():
    """tool_choice none must not produce tool_use blocks"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }],
        "tool_choice": {"type": "none"},
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    content_types = [block.get("type") for block in res.body["content"]]
    assert "tool_use" not in content_types, f"tool_choice none produced tool_use: {content_types}"


def test_anthropic_tool_choice_named_tool_is_enforced():
    """tool_choice on a name restricts the callable tools to that name"""
    server.jinja = True
    server.start()

    # 'unknown_tool' is not in tools: the named choice must be rejected, which proves the
    # name is carried through instead of degrading to a plain 'required'.
    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
        }],
        "tool_choice": {"type": "tool", "name": "unknown_tool"},
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    })

    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "unknown_tool" in res.body["error"]["message"]


def test_anthropic_output_config_effort_invalid():
    """output_config.effort outside the official enum is rejected"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "output_config": {"effort": "turbo"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "output_config.effort" in res.body["error"]["message"]


def test_anthropic_output_config_effort():
    """output_config.effort is accepted and mapped to the local effort"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_output_config_format_json_schema():
    """output_config.format json_schema constrains the response to valid JSON"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 200,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]
                }
            }
        },
        "messages": [
            {"role": "user", "content": "Reply with JSON"}
        ]
    })

    assert res.status_code == 200
    text = next((b["text"] for b in res.body["content"] if b.get("type") == "text"), "")
    assert isinstance(json.loads(text), dict), f"Expected a JSON object, got: {text!r}"


def test_anthropic_thinking_disabled():
    """thinking disabled is accepted and does not emit thinking blocks"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    content_types = [block.get("type") for block in res.body["content"]]
    assert "thinking" not in content_types, f"thinking disabled produced a thinking block: {content_types}"


def test_anthropic_cache_control_breakpoint():
    """cache_control on a system block maps to a local prompt cache breakpoint"""
    server.start()

    payload = {
        "model": "test",
        "max_tokens": 16,
        "system": [
            {
                "type": "text",
                "text": "You are a helpful assistant. " + "Cache me. " * 40,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }

    first = server.make_request("POST", "/v1/messages", data=payload)
    assert first.status_code == 200, f"Expected 200, got {first.status_code}: {first.body}"

    # the breakpoint forces a checkpoint, so the identical follow-up reads the cached prefix
    second = server.make_request("POST", "/v1/messages", data=payload)
    assert second.status_code == 200, f"Expected 200, got {second.status_code}: {second.body}"
    assert second.body["usage"]["cache_read_input_tokens"] > 0, \
        f"Expected a prompt cache hit, got usage {second.body['usage']}"


def test_anthropic_cache_control_invalid():
    """cache_control with an unknown type or ttl is rejected"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 16,
        "system": [
            {
                "type": "text",
                "text": "system",
                "cache_control": {"type": "persistent"}
            }
        ],
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.body}"
    assert "cache_control.type" in res.body["error"]["message"]


def test_anthropic_cache_control_tools_anchor():
    """cache_control on tools[] maps to the tools anchor and warms the prompt cache"""
    server.jinja = True
    server.start()

    payload = {
        "model": "test",
        "max_tokens": 16,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"},
        }],
        "tool_choice": {"type": "none"},
        "messages": [
            {"role": "user", "content": "Hello " + "cache the tools. " * 40}
        ],
    }

    first = server.make_request("POST", "/v1/messages", data=payload)
    assert first.status_code == 200, f"Expected 200, got {first.status_code}: {first.body}"

    second = server.make_request("POST", "/v1/messages", data=payload)
    assert second.status_code == 200, f"Expected 200, got {second.status_code}: {second.body}"
    assert second.body["usage"]["cache_read_input_tokens"] > 0, \
        f"Expected the tools anchor to warm the cache, got usage {second.body['usage']}"


def test_anthropic_cache_control_tool_blocks():
    """cache_control on a tool_use / tool_result block is accepted"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 16,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }],
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                    "cache_control": {"type": "ephemeral"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "sunny",
                    "cache_control": {"type": "ephemeral"},
                }],
            },
        ],
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    assert res.body["type"] == "message"


def test_anthropic_cache_control_ttl_ladder():
    """cache_control.ttl accepts the official 5m/1h values (internal TTL channel)"""
    server.start()

    for ttl in ("5m", "1h"):
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 16,
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant. " + "Cache me. " * 40,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            }],
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert res.status_code == 200, f"Expected 200 for ttl={ttl}, got {res.status_code}: {res.body}"

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 16,
        "system": [{
            "type": "text",
            "text": "system",
            "cache_control": {"type": "ephemeral", "ttl": "30m"},
        }],
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert res.status_code == 400, f"Expected 400 for ttl=30m, got {res.status_code}: {res.body}"
