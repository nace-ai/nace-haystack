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

# Longer messages are cut, so an unexpected response body can't flood logs and tracebacks
MAX_MESSAGE_CHARS = 500

_IMAGE_PATH = re.compile(r"^images\.(\d+)")


def readable_error(error: nace_sdk.NaceError, provider: str, *, image_types: Sequence[str] = ()) -> nace_sdk.NaceError:
    """
    Rewrites an API error so its message says who refused the request and what to change.

    The result is an instance of the same nace-sdk class, so `except nace_sdk.NaceError` (or a narrower class such
    as `nace_sdk.NaceRateLimitError`) still catches it, and its `status`, `body`, `issues`, `request_id` and
    `idempotency_key` are kept. A 404 or 405 names the URL, which is how a wrong base URL shows.

    :param error: The error nace-sdk raised.
    :param provider: Who answered: `"Drex"`, or `"OpenRouter"` on that backend.
    :param image_types: The MIME type of each image sent, in order, so a refused PDF can be named.
    :returns: The readable error, or `error` itself if it isn't an HTTP error response.
    """
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
    for issue in issues:
        refused = _IMAGE_PATH.match(issue.get("path") or "")
        if (
            refused
            and int(refused.group(1)) < len(image_types)
            and image_types[int(refused.group(1))] == "application/pdf"
        ):
            number = int(refused.group(1)) + 1
            text += (
                f"\nImage {number} is a PDF, and Drex reads images. Convert PDF pages to images first, for example "
                "with Haystack's PDFToImageContent."
            )
    if error.status in (404, 405) and error.endpoint:
        # Usually a wrong base URL (`api_base_url` or `NACE_BASE_URL`)
        text += f"\n{error.endpoint} doesn't exist. Check the base URL."
    if error.status == 402:  # noqa: PLR2004
        if provider == "OpenRouter":
            text += f"\nAdd credits at {OPENROUTER_CREDITS_URL}"
        else:
            text += f"\nAdd funds at {BILLING_URL}"
    return text
