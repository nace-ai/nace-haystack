# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
from typing import Any

import nace_sdk
from haystack import default_to_dict, logging
from haystack.utils import Secret

from haystack_integrations.common.nace.client import (
    client_options,
    is_openrouter,
    make_async_client,
    make_client,
    provider_name,
    resolve_base_url,
)
from haystack_integrations.common.nace.concurrency import map_in_threads
from haystack_integrations.common.nace.errors import readable_error
from haystack_integrations.common.nace.images import ImageInput, image_types
from haystack_integrations.common.nace.request import DrexResponse, build_call

logger = logging.getLogger(__name__)

_QUESTION_ID = "verdict"

Outcome = tuple[dict[str, Any], dict[str, Any]]


class _NaceEvaluator:
    """
    Shared client lifecycle and evaluation loop for the Drex evaluators.

    Each item is judged in its own request with one `noul` question; its score is Drex's probability that the
    statement in `instructions` is true. The outputs match Haystack's built-in LLM evaluators.
    """

    def __init__(
        self,
        *,
        instructions: str,
        model: str,
        api_key: Secret,
        api_base_url: str | None,
        timeout: float | None,
        max_retries: int | None,
        max_workers: int,
        raise_on_failure: bool,
    ) -> None:
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.raise_on_failure = raise_on_failure
        self._client: nace_sdk.NaceClient | None = None
        self._async_client: nace_sdk.AsyncNaceClient | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None

    def _client_options(self) -> dict[str, Any]:
        return client_options(self.api_key, self.api_base_url, self.timeout, self.max_retries)

    def warm_up(self) -> None:
        """
        Creates the synchronous Nace client.
        """
        if self._client is None:
            self._client = make_client(self._client_options())

    async def warm_up_async(self) -> None:
        """
        Creates the asynchronous Nace client for the running event loop.
        """
        loop = asyncio.get_running_loop()
        if self._async_client_loop is not None and self._async_client_loop is not loop:
            # An async HTTP client belongs to the event loop it first ran on, so a new `asyncio.run` (a second
            # notebook cell, a second script run) needs a new client. The old one can't be closed from this loop.
            self._async_client = None
        if self._async_client is None:
            self._async_client = make_async_client(self._client_options())
        self._async_client_loop = loop

    def close(self) -> None:
        """
        Releases the synchronous Nace client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Releases the asynchronous Nace client.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_client_loop = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            instructions=self.instructions,
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )

    @staticmethod
    def _check_contexts(contexts: list[list[str]]) -> None:
        for index, context in enumerate(contexts):
            if isinstance(context, str):
                msg = (
                    "`contexts` must be a list of lists of strings, one list per question; "
                    f"item {index} is a single string. Wrap it in a list: [[...]]."
                )
                raise ValueError(msg)

    @staticmethod
    def _check_images(images: list[list[ImageInput]] | None) -> None:
        for index, item_images in enumerate(images or ()):
            if not isinstance(item_images, (list, tuple)):
                msg = (
                    "`images` must hold one list of images per item, such as [[image], []]; "
                    f"item {index} is a single {type(item_images).__name__}. Wrap it in a list."
                )
                raise ValueError(msg)

    @staticmethod
    def _check_lengths(inputs: dict[str, list[Any] | None]) -> None:
        lengths = {name: len(values) for name, values in inputs.items() if values is not None}
        if len(set(lengths.values())) > 1:
            names = ", ".join(f"`{name}` ({length})" for name, length in lengths.items())
            msg = f"The inputs must have the same length: {names}."
            raise ValueError(msg)

    def _calls(
        self, states: list[dict[str, Any]], images: list[list[ImageInput]] | None
    ) -> list[tuple[dict[str, Any], list[ImageInput]]]:
        # Built for every item before any request, so a bad model, image or size fails before anything is billed
        openrouter = is_openrouter(resolve_base_url(self.api_base_url))
        question = {_QUESTION_ID: {"type": "noul", "instructions": self.instructions}}
        calls = []
        for index, state in enumerate(states):
            item_images = list(images[index]) if images is not None else []
            call = build_call(state, question, model=self.model, images=item_images or None, openrouter=openrouter)
            calls.append((call, item_images))
        return calls

    @staticmethod
    def _judged(response: DrexResponse) -> Outcome:
        probability = response.answers[_QUESTION_ID]["noul"]
        meta = {
            "model": response.model,
            "usage": response.usage,
            "request_id": (response.model_extra or {}).get("request_id"),
        }
        return {"score": probability, "noul": probability, "status": "evaluated"}, meta

    def _failed(self, error: nace_sdk.NaceError, item_images: list[ImageInput]) -> Outcome:
        readable = readable_error(
            error, provider_name(resolve_base_url(self.api_base_url)), image_types=image_types(item_images)
        )
        if self.raise_on_failure:
            raise readable from error
        meta = {"model": None, "usage": None, "request_id": getattr(readable, "request_id", None)}
        return {"score": math.nan, "status": "error", "error": str(readable)}, meta

    @staticmethod
    def _summarize(outcomes: list[Outcome]) -> dict[str, Any]:
        results = [result for result, _ in outcomes]
        scores = [result["score"] for result in results]
        valid = [score for score in scores if not math.isnan(score)]
        skipped = len(scores) - len(valid)
        if skipped:
            logger.warning("{skipped} item(s) failed and were excluded from the score.", skipped=skipped)
        return {
            "score": sum(valid) / len(valid) if valid else math.nan,
            "individual_scores": scores,
            "results": results,
            "meta": [meta for _, meta in outcomes],
        }

    def _evaluate(self, states: list[dict[str, Any]], images: list[list[ImageInput]] | None) -> dict[str, Any]:
        calls = self._calls(states, images)
        if not calls:
            return self._summarize([])
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above
        client = self._client

        def judge(call: dict[str, Any], item_images: list[ImageInput]) -> Outcome:
            try:
                return self._judged(client.system_one(**call))
            except nace_sdk.NaceError as error:
                return self._failed(error, item_images)

        outcomes = map_in_threads(lambda pair: judge(*pair), calls, self.max_workers)
        return self._summarize(outcomes)

    async def _evaluate_async(
        self, states: list[dict[str, Any]], images: list[list[ImageInput]] | None
    ) -> dict[str, Any]:
        calls = self._calls(states, images)
        if not calls:
            return self._summarize([])
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above
        client = self._async_client
        semaphore = asyncio.Semaphore(self.max_workers)

        async def judge(call: dict[str, Any], item_images: list[ImageInput]) -> Outcome:
            try:
                async with semaphore:
                    response = await client.system_one(**call)
            except nace_sdk.NaceError as error:
                return self._failed(error, item_images)
            return self._judged(response)

        tasks = [asyncio.create_task(judge(call, item_images)) for call, item_images in calls]
        try:
            outcomes = await asyncio.gather(*tasks)
        except Exception:
            # Only reached when `raise_on_failure` is `True`: cancel the remaining requests before re-raising
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return self._summarize(list(outcomes))
