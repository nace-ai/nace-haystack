# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

import httpx
import nace_sdk
from haystack import logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

try:
    __version__ = version("nace-haystack")
except PackageNotFoundError:  # a source checkout that isn't installed
    __version__ = "0.0.0"

CLIENT_NAME = f"nace-haystack/{__version__}"

# The newest Drex model, used when a component is created without `model`. Bumped in a package release whenever
# Drex ships a new model (see RELEASING.md); every other model id is sent as given and Drex decides.
DEFAULT_MODEL = "drex-max-v1.1"

# Released, pinned versions never gain image input, so this list can't go stale.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0"})

# Drex may take up to 55 seconds to answer, so shorter timeouts can cut off a valid request.
MIN_TIMEOUT = 60.0

# The longest wait between retries, in seconds, even when the server asks for a longer one
MAX_RETRY_WAIT = 60.0


class CappedRetryPolicy(nace_sdk.RetryPolicy):
    """
    nace-sdk's retry policy, with every wait capped at `MAX_RETRY_WAIT`.

    The SDK honours `retry-after-ms` however long it is, so a misbehaving server or gateway could stall a pipeline
    for hours. This overrides the SDK's private `_wait` until the SDK caps waits itself.
    """

    def _wait(self, attempt: int, error: BaseException) -> float:
        return min(super()._wait(attempt, error), MAX_RETRY_WAIT)


def is_text_only(model: str) -> bool:
    """
    Whether `model` is a released Drex version known to read text only.

    :param model: A Drex model id, optionally with a provider prefix such as `nace-ai/` and an OpenRouter variant
        suffix such as `:floor`.
    :returns: `True` for `drex-v1.5` and `drex-v2.0`, `False` for every other id.
    """
    return model.rsplit("/", 1)[-1].lstrip("~").partition(":")[0] in TEXT_ONLY_MODELS


def is_openrouter(base_url: str | None) -> bool:
    """
    Whether `base_url` points to OpenRouter, which takes a different request format.

    :param base_url: The API base URL, or `None` for the default.
    :returns: `True` if the host is openrouter.ai.
    """
    return bool(base_url) and (urlsplit(str(base_url)).hostname or "").endswith("openrouter.ai")


def provider_name(base_url: str | None) -> str:
    """
    Names who answers requests sent to `base_url`, for error messages.

    :param base_url: The API base URL, or `None` for the default.
    :returns: `"OpenRouter"` or `"Drex"`.
    """
    return "OpenRouter" if is_openrouter(base_url) else "Drex"


def resolve_base_url(api_base_url: str | None) -> str | None:
    """
    Resolves the API base URL from the argument or the `NACE_BASE_URL` environment variable.

    :param api_base_url: The base URL passed to the component, or `None`.
    :returns: The base URL, or `None` for the SDK default. A blank value (common in `.env` files and CI) means the
        default. OpenRouter's documented `.../api/v1` is trimmed to `.../api`, because the SDK appends `/v1/...`.
    """
    base_url = api_base_url if api_base_url is not None else os.environ.get("NACE_BASE_URL")
    base_url = (base_url or "").strip() or None
    if base_url and is_openrouter(base_url):
        base_url = base_url.rstrip("/").removesuffix("/v1")
    return base_url


def client_options(
    api_key: Secret, api_base_url: str | None, timeout: float | None, max_retries: int | None
) -> dict[str, Any]:
    """
    Builds the keyword arguments for `nace_sdk.NaceClient` and `nace_sdk.AsyncNaceClient`.

    :param api_key: The Nace API key.
    :param api_base_url: The base URL passed to the component, or `None`.
    :param timeout: Timeout in seconds for each request attempt. `None` means 60 seconds.
    :param max_retries: Maximum number of retries after a failed attempt. `None` means the SDK default of 2.
    :returns: The client keyword arguments.
    """
    base_url = resolve_base_url(api_base_url)
    if timeout is None:
        timeout = MIN_TIMEOUT
    elif timeout < MIN_TIMEOUT:
        logger.warning(
            "A timeout of {timeout} s can cut off Drex, which may take up to 55 s to answer. Use at least {minimum} s.",
            timeout=timeout,
            minimum=MIN_TIMEOUT,
        )
    headers = {}
    if is_openrouter(base_url):
        headers = {"HTTP-Referer": "https://nace.ai", "X-OpenRouter-Title": "Drex for Haystack"}
    # The SDK's retry budget defaults to 30 seconds in total, which would skip retries after a long attempt.
    # `timeout` already bounds each attempt, so the budget is disabled.
    retry_kwargs: dict[str, Any] = {"timeout": None}
    if max_retries is not None:
        retry_kwargs["max_retries"] = max_retries
    return {
        "api_key": api_key.resolve_value(),
        "base_url": base_url,
        "timeout": timeout,
        "retry": CappedRetryPolicy(**retry_kwargs),
        "headers": headers,
        "client_info": CLIENT_NAME,
    }


def _set_user_agent(request: httpx.Request) -> None:
    # The SDK sends its own User-Agent; ours names this integration so traffic can be measured per channel.
    request.headers["User-Agent"] = CLIENT_NAME


async def _set_user_agent_async(request: httpx.Request) -> None:
    _set_user_agent(request)


def make_client(options: dict[str, Any]) -> nace_sdk.NaceClient:
    """
    Creates a synchronous Nace client that identifies this integration on every request.

    :param options: Keyword arguments from `client_options`, plus an optional `transport` for tests.
    :returns: The client.
    """
    options = dict(options)
    http_client = httpx.Client(
        timeout=options["timeout"], transport=options.pop("transport", None), event_hooks={"request": [_set_user_agent]}
    )
    return nace_sdk.NaceClient(**options, http_client=http_client)


def make_async_client(options: dict[str, Any]) -> nace_sdk.AsyncNaceClient:
    """
    Creates an asynchronous Nace client that identifies this integration on every request.

    :param options: Keyword arguments from `client_options`, plus an optional `transport` for tests.
    :returns: The client.
    """
    options = dict(options)
    http_client = httpx.AsyncClient(
        timeout=options["timeout"],
        transport=options.pop("transport", None),
        event_hooks={"request": [_set_user_agent_async]},
    )
    return nace_sdk.AsyncNaceClient(**options, http_client=http_client)
