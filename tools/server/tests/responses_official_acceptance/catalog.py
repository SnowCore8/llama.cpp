"""Official OpenAI Responses + adjacent stable surfaces catalog (from openai SDK)."""

from __future__ import annotations

from dataclasses import dataclass

from openai.resources.responses.responses import Responses
from openai.types.completion_create_params import CompletionCreateParamsBase
from openai.types.responses.response_create_params import ResponseCreateParamsBase


@dataclass(frozen=True)
class OfficialEndpoint:
    method: str
    path_template: str
    sdk_name: str
    kind: str  # http | websocket | sdk_helper


def sdk_response_methods() -> list[str]:
    names = [
        "create",
        "retrieve",
        "delete",
        "cancel",
        "compact",
        "input_items",
        "input_tokens",
        "stream",
        "parse",
        "connect",
    ]
    return [n for n in names if hasattr(Responses, n)]


def official_endpoints() -> list[OfficialEndpoint]:
    """Stable OpenAI surfaces: Responses (+ helpers) + Completions + Models."""
    eps = [
        OfficialEndpoint("POST", "/v1/responses", "responses.create", "http"),
        OfficialEndpoint("GET", "/v1/responses/{response_id}", "responses.retrieve", "http"),
        OfficialEndpoint("DELETE", "/v1/responses/{response_id}", "responses.delete", "http"),
        OfficialEndpoint(
            "POST", "/v1/responses/{response_id}/cancel", "responses.cancel", "http"
        ),
        OfficialEndpoint("POST", "/v1/responses/compact", "responses.compact", "http"),
        OfficialEndpoint(
            "GET",
            "/v1/responses/{response_id}/input_items",
            "responses.input_items",
            "http",
        ),
        OfficialEndpoint(
            "POST", "/v1/responses/input_tokens", "responses.input_tokens", "http"
        ),
        OfficialEndpoint("POST", "/v1/completions", "completions.create", "http"),
        OfficialEndpoint("GET", "/v1/models", "models.list", "http"),
        OfficialEndpoint("SDK", "client.responses.parse", "parse", "sdk_helper"),
        OfficialEndpoint("SDK", "client.responses.stream", "stream", "sdk_helper"),
    ]
    if hasattr(Responses, "connect"):
        eps.append(
            OfficialEndpoint("WS", "/v1/responses (connect)", "connect", "websocket")
        )
    return eps


def create_param_names() -> list[str]:
    return sorted(ResponseCreateParamsBase.__annotations__.keys())


def completion_param_names() -> list[str]:
    return sorted(CompletionCreateParamsBase.__annotations__.keys())


CORE_STREAM_EVENTS = [
    "response.created",
    "response.in_progress",
    "response.completed",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
]

CONDITIONAL_STREAM_EVENTS = [
    "response.failed",
    "response.incomplete",
    "response.reasoning_text.delta",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "error",
]
