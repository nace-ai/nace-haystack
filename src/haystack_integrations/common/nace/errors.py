# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Any

import nace_sdk

BILLING_URL = "https://console.nace.ai/dashboard/billing"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/settings/credits"

# Budget for error.message, including provider prefix and recovery guidance.
# The SDK renders status/request ID separately; structured body and issues remain available.
MAX_MESSAGE_CHARS = 500

_IMAGE_PATH = re.compile(r"^images\.(\d+)")

# What a value the error text held is replaced with, so a reader can tell something was removed.
_REDACTED = "[redacted]"


def redact_api_key(error: nace_sdk.NaceError, api_key: str | None) -> nace_sdk.NaceError:
    """
    Returns `error` with the API key value removed from its message and body.

    A gateway or proxy can answer a failed request by echoing the `Authorization` value it got, which
    would put the key into logs and raised errors. The returned copy keeps the error's class, status,
    headers, endpoint and idempotency key, so branching on them still works. An error that is not an
    HTTP error response, that holds the key nowhere, or a key that is unknown, gives the error back
    as it is.

    :param error: The error nace-sdk raised.
    :param api_key: The API key value the request carried, or `None` when it is unknown.
    :returns: The error, or a copy of it without the API key value.
    """
    if (
        not api_key
        or not isinstance(error, nace_sdk.NaceAPIError)
        or isinstance(error, nace_sdk.NaceAPIResponseValidationError)
    ):
        return error
    message = error.message
    if isinstance(message, str) and api_key in message:
        message = message.replace(api_key, _REDACTED)
    body = _without_api_key(error.body, api_key)
    if message == error.message and body == error.body:
        return error
    redacted = type(error)(error.status, body, error.headers, message=message, endpoint=error.endpoint)
    redacted.idempotency_key = error.idempotency_key
    return redacted


def _without_api_key(value: Any, api_key: str) -> Any:
    # The key value can sit at any depth of a JSON error body, so every string in it is cleaned.
    if isinstance(value, str):
        return value.replace(api_key, _REDACTED) if api_key in value else value
    if isinstance(value, dict):
        return {name: _without_api_key(item, api_key) for name, item in value.items()}
    if isinstance(value, list):
        return [_without_api_key(item, api_key) for item in value]
    return value


def readable_error(
    error: nace_sdk.NaceError,
    provider: str,
    *,
    image_types: Sequence[str] = (),
    api_key: str | None = None,
) -> nace_sdk.NaceError:
    """
    Rewrites an API error so its message says who refused the request and what to change.

    The result is an instance of the same nace-sdk class, so `except nace_sdk.NaceError` (or a narrower class such
    as `nace_sdk.NaceRateLimitError`) still catches it, and its `status`, `body`, `issues`, `request_id` and
    `idempotency_key` are kept. A 404 or 405 names the URL, which is how a wrong base URL shows. When the API key
    value is given, it is first removed from the error's message and body, so an error that echoes the
    `Authorization` value cannot leak the key.

    :param error: The error nace-sdk raised.
    :param provider: Who answered: `"Drex"`, or `"OpenRouter"` on that backend.
    :param image_types: The MIME type of each image sent, in order, so a refused PDF can be named.
    :param api_key: The API key value the request carried, or `None` to leave the error as it is.
    :returns: The readable error, or `error` itself if it isn't an HTTP error response.
    """
    error = redact_api_key(error, api_key)
    if not isinstance(error, nace_sdk.NaceAPIError) or isinstance(error, nace_sdk.NaceAPIResponseValidationError):
        return error
    message, found = _unwrap(error.message) if error.message else (None, [])
    if message:
        message = _shorten(message, provider, error.status)
    issues = error.issues or found
    # nace-sdk prints errors as "<status> <message> (request_id=...)"
    readable = type(error)(
        error.status,
        error.body,
        error.headers,
        message=f"from {provider}: {_describe(error, provider, message, issues, image_types)}",
    )
    readable.idempotency_key = error.idempotency_key
    return readable


def _unwrap(message: str) -> tuple[str, list[dict[str, str]]]:
    # Gateways such as OpenRouter pass upstream errors through as text like `HTTP 422: {"detail": "..."}` or a JSON
    # list of schema issues. Return a short message and any `{"path", "message"}` issues found in it.
    text = message.strip()
    wrapped = re.match(r"^HTTP \d{3}: (.*)$", text, re.DOTALL)
    if wrapped:
        text = wrapped.group(1).strip()
    try:
        data: Any = json.loads(text)
    except ValueError:
        return text, []
    if isinstance(data, dict):
        inner = data.get("detail") or data.get("message") or data.get("error")
        if isinstance(inner, str):
            return inner, []
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"], []
        data = inner
    if isinstance(data, list):
        issues = []
        for item in data:
            if isinstance(item, dict) and item.get("message"):
                path = item.get("path", "")
                if isinstance(path, list):
                    path = ".".join(str(part) for part in path)
                issues.append({"path": str(path), "message": str(item["message"])})
        if issues:
            return "the request is invalid", issues
    return text, []


def _clip(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[: max(0, budget - 1)] + "…"


def _shorten(message: str, provider: str, status: int) -> str:
    # Gateways and proxies answer some failures with an HTML page, which says nothing useful in a Python error
    if message.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        return f"{provider} returned an HTML error page (HTTP {status})"
    if len(message) > MAX_MESSAGE_CHARS:
        return message[:MAX_MESSAGE_CHARS] + "…"
    return message


def _describe(
    error: nace_sdk.NaceAPIError,
    provider: str,
    message: str | None,
    issues: list[dict[str, str]],
    image_types: Sequence[str],
) -> str:
    if error.status == 413:  # noqa: PLR2004
        # Often an HTML page from the gateway rather than a Drex error
        if not image_types:
            return "the request is too large. Shorten the text."
        text = (
            "the request is too large. Send fewer or smaller images, or pass big images as files uploaded to Drex "
            "with nace-sdk."
        )
        if "application/pdf" in image_types:
            text += (
                "\nA PDF can't be shrunk to fit: convert its pages to images first, for example with Haystack's "
                "PDFToImageContent."
            )
        return text
    text = message or "the request failed"
    lines = [f"{issue.get('path') or '(request)'}: {issue['message']}" for issue in issues if issue.get("message")]
    # Drex's top-level message often repeats a single issue; don't print it twice
    repeated = len(issues) == 1 and (issues[0].get("message") == message or lines == [message])
    if lines and not repeated:
        text += "\n" + "\n".join(f"  {line}" for line in lines)
    guidance = ""
    for issue in issues:
        refused = _IMAGE_PATH.match(issue.get("path") or "")
        if (
            refused
            and int(refused.group(1)) < len(image_types)
            and image_types[int(refused.group(1))] == "application/pdf"
        ):
            number = int(refused.group(1)) + 1
            guidance += (
                f"\nImage {number} is a PDF, and Drex reads images. Convert PDF pages to images first, for example "
                "with Haystack's PDFToImageContent."
            )
    if error.status in (404, 405) and error.endpoint:
        # Usually a wrong base URL (`api_base_url` or `NACE_BASE_URL`)
        guidance = f"\n{_clip(error.endpoint, 200)} doesn't exist. Check the base URL." + guidance
    if error.status == 402:  # noqa: PLR2004
        if provider == "OpenRouter":
            guidance = f"\nAdd credits at {OPENROUTER_CREDITS_URL}" + guidance
        else:
            guidance = f"\nAdd funds at {BILLING_URL}" + guidance
    guidance = _clip(guidance, 350)
    budget = MAX_MESSAGE_CHARS - len(f"from {provider}: ") - len(guidance)
    return _clip(text, budget) + guidance
