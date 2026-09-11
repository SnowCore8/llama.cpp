"""SDK schema validators for official Response / stream events."""

from __future__ import annotations

from typing import Any

from openai.types.responses import Response
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_content_part_done_event import (
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_failed_event import ResponseFailedEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_in_progress_event import ResponseInProgressEvent
from openai.types.responses.response_incomplete_event import ResponseIncompleteEvent
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent

try:
    from openai.types.responses.response_reasoning_text_delta_event import (
        ResponseReasoningTextDeltaEvent,
    )
    from openai.types.responses.response_reasoning_text_done_event import (
        ResponseReasoningTextDoneEvent,
    )
    from openai.types.responses.response_reasoning_summary_part_added_event import (
        ResponseReasoningSummaryPartAddedEvent,
    )
    from openai.types.responses.response_reasoning_summary_part_done_event import (
        ResponseReasoningSummaryPartDoneEvent,
    )
    from openai.types.responses.response_reasoning_summary_text_delta_event import (
        ResponseReasoningSummaryTextDeltaEvent,
    )
    from openai.types.responses.response_reasoning_summary_text_done_event import (
        ResponseReasoningSummaryTextDoneEvent,
    )
except Exception:  # pragma: no cover - older SDK
    ResponseReasoningTextDeltaEvent = None  # type: ignore
    ResponseReasoningTextDoneEvent = None  # type: ignore
    ResponseReasoningSummaryPartAddedEvent = None  # type: ignore
    ResponseReasoningSummaryPartDoneEvent = None  # type: ignore
    ResponseReasoningSummaryTextDeltaEvent = None  # type: ignore
    ResponseReasoningSummaryTextDoneEvent = None  # type: ignore

EVENT_VALIDATORS = {
    "response.created": ResponseCreatedEvent,
    "response.in_progress": ResponseInProgressEvent,
    "response.completed": ResponseCompletedEvent,
    "response.incomplete": ResponseIncompleteEvent,
    "response.failed": ResponseFailedEvent,
    "error": ResponseErrorEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.output_text.done": ResponseTextDoneEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
}
if ResponseReasoningTextDeltaEvent is not None:
    EVENT_VALIDATORS["response.reasoning_text.delta"] = ResponseReasoningTextDeltaEvent
if ResponseReasoningTextDoneEvent is not None:
    EVENT_VALIDATORS["response.reasoning_text.done"] = ResponseReasoningTextDoneEvent
if ResponseReasoningSummaryPartAddedEvent is not None:
    EVENT_VALIDATORS["response.reasoning_summary_part.added"] = ResponseReasoningSummaryPartAddedEvent
if ResponseReasoningSummaryPartDoneEvent is not None:
    EVENT_VALIDATORS["response.reasoning_summary_part.done"] = ResponseReasoningSummaryPartDoneEvent
if ResponseReasoningSummaryTextDeltaEvent is not None:
    EVENT_VALIDATORS["response.reasoning_summary_text.delta"] = ResponseReasoningSummaryTextDeltaEvent
if ResponseReasoningSummaryTextDoneEvent is not None:
    EVENT_VALIDATORS["response.reasoning_summary_text.done"] = ResponseReasoningSummaryTextDoneEvent



def validate_response(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        else:
            return False, f"expected dict or Response, got {type(obj).__name__}"
    required = [k for k, v in Response.model_fields.items() if v.is_required()]
    missing = [k for k in required if k not in obj]
    try:
        Response.model_validate(obj)
        sdk_ok, sdk_err = True, ""
    except Exception as e:
        sdk_ok, sdk_err = False, str(e)
    if missing or not sdk_ok:
        return False, f"missing={missing}; sdk={sdk_err[:240]}"
    return True, "ok"


def validate_event(event_type: str, obj: dict[str, Any]) -> tuple[bool, str]:
    cls = EVENT_VALIDATORS.get(event_type)
    if cls is None:
        return True, "no_sdk_validator"
    try:
        cls.model_validate(obj)
        return True, "ok"
    except Exception as e:
        return False, str(e)[:240]
