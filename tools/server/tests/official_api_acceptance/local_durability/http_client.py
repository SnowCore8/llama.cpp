"""Reuse Responses HTTP client helpers."""

from __future__ import annotations

from official_api_acceptance.responses.http_client import (  # noqa: F401
    ResponsesHttpClient,
    output_text,
    parse_sse,
)

HttpClient = ResponsesHttpClient
