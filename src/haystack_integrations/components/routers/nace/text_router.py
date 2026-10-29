# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

# Mirrors deepset's TypeSafeTextRouter (typesafe-haystack) so a Jev pipeline switches to Drex by changing the import,
# the class name and the API key.
from typing import Any

import nace_sdk
from haystack import component, default_to_dict
from haystack.utils import Secret

from haystack_integrations.common.nace.client import (
    DEFAULT_MODEL,
    client_options,
    is_openrouter,
    make_async_client,
    make_client,
    provider_name,
    resolve_base_url,
)
from haystack_integrations.common.nace.errors import readable_error
from haystack_integrations.common.nace.images import ImageInput, image_types
from haystack_integrations.common.nace.request import DrexResponse, build_call

_LOW_CONFIDENCE = "low_confidence"
_PROBABILITIES = "probabilities"
_QUESTION_ID = "route"


@component
class NaceTextRouter:
    """
    Routes a text to the connection of the label a Drex decision model picks for it.

    The model answers a single `choice` question over `labels` and returns calibrated probabilities. The component
    has one output per label. When `min_confidence` is set, it also has a `low_confidence` output that receives the
    text whenever the answer's `confidence` is below the threshold, so uncertain texts can go to a fallback such as
    an LLM or a human. With `include_probabilities=True`, a `probabilities` output carries the full answer, so
    thresholds stay in your pipeline.

    Images can be routed too: pass them to `run` and the default model, `drex-max-v1.1`, reads them alongside the text,
    as "Image 1", "Image 2" and so on. Images too big for one request are shrunk to fit.

    The component also works with OpenRouter and other TypeSafe-compatible hosts: set `api_base_url` to the server
    and `model` to one of its models.

    ### Usage example

    ```python
    from haystack_integrations.components.routers.nace import NaceTextRouter

    router = NaceTextRouter(
        labels={
            "billing": "Payments, invoices, refunds and charges",
            "technical": "Bugs, crashes and errors in the product",
        },
        instructions="Which team should handle this support request?",
        min_confidence=0.6,
    )

    print(router.run(text="I was charged twice for my subscription."))
    # {'billing': 'I was charged twice for my subscription.'}
    ```
    """

    def __init__(
        self,
        labels: list[str] | dict[str, str | None],
        *,
        instructions: str = "Which category does this text belong to?",
        model: str = DEFAULT_MODEL,
        api_key: Secret = Secret.from_env_var("NACE_API_KEY"),
        api_base_url: str | None = None,
        min_confidence: float | None = None,
        include_probabilities: bool = False,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Creates a NaceTextRouter.

        :param labels:
            The labels to route between, each becoming an output connection. Pass a dict of label to description to
            tell the model what each label means.
        :param instructions:
            The question the model answers by picking one of the labels.
        :param model:
            The Drex model to use. Defaults to the newest Drex model, which reads images; pin a model such as
            `"drex-max-v1.1"` or `"drex-v2.0"` in production. On OpenRouter, use its model IDs, such as
            `"nace-ai/drex-max"`.
        :param api_key:
            The Nace API key.
        :param api_base_url:
            Base URL of the API. If `None`, the `NACE_BASE_URL` environment variable is used, or
            `https://console.nace.ai` if it is unset.
        :param min_confidence:
            Threshold between 0 and 1 for the answer's `confidence`. Texts below it are sent to the `low_confidence`
            output. If `None`, every text goes to a label output.
        :param include_probabilities:
            If `True`, a `probabilities` output also receives the full answer: the picked label, its `confidence` and
            every label's probability.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, 60 seconds are used, because Drex may take up to
            55 seconds to answer.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :raises ValueError:
            If `labels` is empty, contains a reserved output name or a described label named `type`, or if
            `min_confidence` is outside 0 to 1.
        """
        if not labels:
            msg = "`labels` must contain at least one label."
            raise ValueError(msg)
        if _LOW_CONFIDENCE in labels:
            msg = f"`{_LOW_CONFIDENCE}` is reserved for the low confidence output and can't be used as a label."
            raise ValueError(msg)
        if include_probabilities and _PROBABILITIES in labels:
            msg = f"`{_PROBABILITIES}` is reserved for the probabilities output and can't be used as a label."
            raise ValueError(msg)
        if isinstance(labels, dict) and "type" in labels:
            # Haystack reads a saved dict with a "type" key as a serialized object, so the pipeline couldn't be loaded
            msg = (
                "A label named `type` can't have a description, because a saved pipeline couldn't be loaded. "
                "Rename the label, or pass `labels` as a list."
            )
            raise ValueError(msg)
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            msg = f"`min_confidence` must be between 0 and 1, got {min_confidence}."
            raise ValueError(msg)

        self.labels = labels
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.min_confidence = min_confidence
        self.include_probabilities = include_probabilities
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: nace_sdk.NaceClient | None = None
        self._async_client: nace_sdk.AsyncNaceClient | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None

        # Drex takes choice criteria as a dict of label to description
        criteria = labels if isinstance(labels, dict) else dict.fromkeys(labels)
        self._question = nace_sdk.Choice(instructions=instructions, criteria=criteria)

        outputs: dict[str, Any] = dict.fromkeys(labels, str)
        if min_confidence is not None:
            outputs[_LOW_CONFIDENCE] = str
        if include_probabilities:
            outputs[_PROBABILITIES] = dict[str, Any]
        component.set_output_types(self, **outputs)

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
            labels=self.labels,
            instructions=self.instructions,
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            min_confidence=self.min_confidence,
            include_probabilities=self.include_probabilities,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _call(self, text: str, images: list[ImageInput] | None) -> dict[str, Any]:
        if not isinstance(text, str):
            msg = "NaceTextRouter expects a str as input."
            raise TypeError(msg)
        return build_call(
            text,
            {_QUESTION_ID: self._question},
            model=self.model,
            images=images,
            openrouter=is_openrouter(resolve_base_url(self.api_base_url)),
        )

    def _error(self, error: nace_sdk.NaceError, images: list[ImageInput] | None) -> nace_sdk.NaceError:
        return readable_error(
            error, provider_name(resolve_base_url(self.api_base_url)), image_types=image_types(images or [])
        )

    def _route(self, text: str, response: DrexResponse) -> dict[str, Any]:
        """
        Picks the output for the text from the Drex response.

        :param text:
            The routed text.
        :param response:
            The Drex response for the text.
        :returns:
            A dictionary mapping `low_confidence` to the text if `min_confidence` is set and the answer's `confidence`
            is below it, otherwise mapping the picked label to the text. With `include_probabilities`, it also maps
            `probabilities` to the full answer.
        """
        answer = response.answers[_QUESTION_ID]
        if answer["choice"] not in self._question.criteria:
            msg = (
                f"{provider_name(resolve_base_url(self.api_base_url))} answered {answer['choice']!r}, which isn't one "
                f"of the labels {list(self._question.criteria)}."
            )
            raise ValueError(msg)
        if self.min_confidence is not None and answer["confidence"] < self.min_confidence:
            routed: dict[str, Any] = {_LOW_CONFIDENCE: text}
        else:
            routed = {answer["choice"]: text}
        if self.include_probabilities:
            routed[_PROBABILITIES] = answer
        return routed

    def run(self, text: str, images: list[ImageInput] | None = None) -> dict[str, Any]:
        """
        Routes the text to the output of the label the model picks.

        :param text:
            The text to route.
        :param images:
            Optional images the model reads with the text, referred to as "Image 1", "Image 2" and so on:
            `ImageContent`, `ByteStream`, or files uploaded to Drex with nace-sdk.
        :returns:
            A dictionary with the picked label or `low_confidence` mapped to the input text, and `probabilities`
            mapped to the full answer if `include_probabilities` is `True`.
        :raises TypeError:
            If `text` is not a string.
        :raises ValueError:
            If images are passed to a model that reads text only.
        """
        call = self._call(text, images)
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        try:
            response = self._client.system_one(**call)
        except nace_sdk.NaceError as error:
            raise self._error(error, images) from error
        return self._route(text=text, response=response)

    async def run_async(self, text: str, images: list[ImageInput] | None = None) -> dict[str, Any]:
        """
        Asynchronously routes the text to the output of the label the model picks.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param text:
            The text to route.
        :param images:
            Optional images the model reads with the text, referred to as "Image 1", "Image 2" and so on.
        :returns:
            A dictionary with the picked label or `low_confidence` mapped to the input text, and `probabilities`
            mapped to the full answer if `include_probabilities` is `True`.
        :raises TypeError:
            If `text` is not a string.
        :raises ValueError:
            If images are passed to a model that reads text only.
        """
        call = self._call(text, images)
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        try:
            response = await self._async_client.system_one(**call)
        except nace_sdk.NaceError as error:
            raise self._error(error, images) from error
        return self._route(text=text, response=response)
