"""Official OpenAI Chat Completions + Models catalog (from openai SDK)."""

from __future__ import annotations

from dataclasses import dataclass

from openai.resources.chat.completions import Completions
from openai.types.chat.completion_create_params import CompletionCreateParamsBase


@dataclass(frozen=True)
class OfficialEndpoint:
    method: str
    path_template: str
    sdk_name: str
    kind: str  # http | sdk_helper


def sdk_chat_methods() -> list[str]:
    names = ["create", "stream", "parse", "retrieve", "list", "update", "delete"]
    return [n for n in names if hasattr(Completions, n)]


def official_endpoints() -> list[OfficialEndpoint]:
    """Stable OpenAI Chat Completions surface (+ Models + store CRUD)."""
    eps = [
        OfficialEndpoint("POST", "/v1/chat/completions", "chat.completions.create", "http"),
        OfficialEndpoint(
            "GET", "/v1/chat/completions", "chat.completions.list", "http"
        ),
        OfficialEndpoint(
            "GET",
            "/v1/chat/completions/{completion_id}",
            "chat.completions.retrieve",
            "http",
        ),
        OfficialEndpoint(
            "POST",
            "/v1/chat/completions/{completion_id}",
            "chat.completions.update",
            "http",
        ),
        OfficialEndpoint(
            "DELETE",
            "/v1/chat/completions/{completion_id}",
            "chat.completions.delete",
            "http",
        ),
        OfficialEndpoint(
            "POST",
            "/v1/chat/completions/input_tokens",
            "chat.completions.input_tokens",
            "http",
        ),
        OfficialEndpoint("GET", "/v1/models", "models.list", "http"),
        OfficialEndpoint("SDK", "client.chat.completions.stream", "stream", "sdk_helper"),
        OfficialEndpoint("SDK", "client.chat.completions.parse", "parse", "sdk_helper"),
    ]
    return eps


def create_param_names() -> list[str]:
    return sorted(CompletionCreateParamsBase.__annotations__.keys())


CORE_STREAM_CHECKS = [
    "content_type_text_event_stream",
    "chunk.object_chat_completion_chunk",
    "delta.role",
    "delta.content",
    "finish_reason",
    "stream_done_sentinel",
    "chunk_sdk_validate",
    "chunk.obfuscation.default",
    "chunk.obfuscation.disabled",
    "chunk.usage.null",
]

CONDITIONAL_STREAM_CHECKS = [
    "delta.tool_calls",
    "delta.reasoning_content",
    "choice.logprobs",
    "stream_error_mid_generation",
    "usage.final_chunk",
]
