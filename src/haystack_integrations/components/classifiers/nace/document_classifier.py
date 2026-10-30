# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

# Mirrors deepset's TypeSafeDocumentClassifier (typesafe-haystack) so a Jev pipeline switches to Drex by changing the
# import, the class name and the API key.

import asyncio
from dataclasses import replace
from typing import Any

import nace_sdk
from haystack import Document, component, default_to_dict, logging
from haystack.utils import Secret

from haystack_integrations.common.nace.client import (
    DEFAULT_MODEL,
    bind_credentials,
    client_options,
    is_openrouter,
    make_async_client,
    make_client,
    provider_name,
    resolve_base_url,
)
from haystack_integrations.common.nace.concurrency import map_in_threads
from haystack_integrations.common.nace.errors import readable_error
from haystack_integrations.common.nace.request import DrexResponse, Question, build_call, validate_questions

logger = logging.getLogger(__name__)

_ERROR_FIELD = "classification_error"


@component
class NaceDocumentClassifier:
    """
    Answers typed questions about each document with a Drex decision model and stores the answers in its metadata.

    Drex models don't generate text: they answer every question in one request and return calibrated probabilities.
    Each question has a type, such as:
    - `choice`: picks one label from `criteria`, a dict of label to description (or `None`).
    - `score`: rates the text on the ordered levels in `criteria`, a list of level descriptions.
    - `noul`: returns the probability that the yes/no question in `instructions` is true.

    Answers are stored under `meta[metadata_field]`, keyed by question ID, exactly as Drex returns them, with every
    probability. Questions can be dicts or nace-sdk `Choice`, `Score` and `Noul` objects. See the
    [Drex documentation](https://console.nace.ai/docs) for the full question and answer format.

    Documents without text to classify are returned in `failed_documents` with the reason in
    `meta["classification_error"]`. So are documents whose request failed, unless `raise_on_failure` is `True`.

    The component also works with OpenRouter and other TypeSafe-compatible hosts: set `api_base_url` to the server
    and `model` to one of its models.

    ### Usage example

    ```python
    from haystack import Document

    from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier

    classifier = NaceDocumentClassifier(
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {"billing": None, "technical": None, "sales": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"],
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        }
    )

    result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
    print(result["documents"][0].meta["nace"])
    # {'department': {'type': 'choice', 'choice': 'billing', 'confidence': 0.97,
    #                 'probabilities': {'billing': 0.98, 'technical': 0.01, 'sales': 0.01}},
    #  'urgency': {'type': 'score', 'score': 1.93, 'confidence': 0.94,
    #              'legend': {'0': 'can wait', '1': 'this week', '2': 'today'},
    #              'probabilities': {'0': 0.01, '1': 0.05, '2': 0.94}},
    #  'refund': {'type': 'noul', 'noul': 0.96}}
    print(result["failed_documents"])
    # []
    ```
    """

    def __init__(
        self,
        questions: dict[str, Question],
        *,
        model: str = DEFAULT_MODEL,
        api_key: Secret | None = None,
        api_base_url: str | None = None,
        classification_field: str | None = None,
        metadata_field: str = "nace",
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates a NaceDocumentClassifier.

        :param questions:
            Questions to answer for every document, keyed by question ID. Each value is a dict or a nace-sdk
            `Choice`, `Score` or `Noul` object with a `type`, `instructions`, and for `choice` and `score` questions,
            `criteria`. Question types Drex adds later can be passed as dicts.
        :param model:
            The Drex model to use. Defaults to the newest Drex model; pin a model such as `"drex-max-v1.1"` or
            `"drex-v2.0"` in production. `drex-v2.0` reads text only and bills fewer tokens on short texts.
        :param api_key:
            A provider key. Defaults to `NACE_API_KEY` on Nace production origins; required for custom hosts.
        :param api_base_url:
            Base URL of the API. If `None`, the `NACE_BASE_URL` environment variable is used, or
            `https://console.nace.ai` if it is unset.
        :param classification_field:
            Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
        :param metadata_field:
            Name of the metadata field the answers are written to.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, 60 seconds are used, because Drex may take up to
            55 seconds to answer.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param max_workers:
            Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
            Free accounts allow 8 requests in flight and paid accounts 16.
        :param raise_on_failure:
            If `True`, the first failed request raises its error. If `False`, documents whose request failed are
            returned in `failed_documents`.
        :raises ValueError:
            If `questions` is empty, a question has no `type`, a `choice` or `score` question has no `criteria`, or
            `max_workers` is smaller than 1.
        """
        validate_questions(questions)
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)

        self.questions = questions
        self.model = model
        self.api_key, self.api_base_url = bind_credentials(api_key, api_base_url)
        self.classification_field = classification_field
        self.metadata_field = metadata_field
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
            questions={
                question_id: question.model_dump() if hasattr(question, "model_dump") else question
                for question_id, question in self.questions.items()
            },
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            classification_field=self.classification_field,
            metadata_field=self.metadata_field,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )

    def _text(self, document: Document) -> Any:
        return document.content if self.classification_field is None else document.meta.get(self.classification_field)

    def _call(self, text: Any) -> dict[str, Any]:
        base_url = resolve_base_url(self.api_base_url)
        return build_call(text, self.questions, model=self.model, images=None, openrouter=is_openrouter(base_url))

    def _succeed(self, document: Document, response: DrexResponse) -> tuple[Document, bool]:
        """
        Adds the answers from a Drex response to a copy of the document.

        :param document:
            The classified document.
        :param response:
            The Drex response for the document.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and no `classification_error`, and
            `True` to mark it as classified.
        """
        # Answers are the JSON Drex sent, so they stay serializable and keep fields added in later Drex releases
        meta = {key: value for key, value in document.meta.items() if key != _ERROR_FIELD}
        return replace(document, meta={**meta, self.metadata_field: response.answers}), True

    def _fail(self, document: Document, error: nace_sdk.NaceError) -> tuple[Document, bool]:
        """
        Records a failed request on a copy of the document, or raises it.

        :param document:
            The document whose request failed.
        :param error:
            The error nace-sdk raised.
        :returns:
            A copy of the document with the readable error under `meta["classification_error"]`, and `False`.
        :raises nace_sdk.NaceError:
            The readable error, if `raise_on_failure` is `True`.
        """
        readable = readable_error(error, provider_name(resolve_base_url(self.api_base_url)))
        if self.raise_on_failure:
            raise readable from error
        logger.warning("Classifying document {document_id} failed: {error}", document_id=document.id, error=readable)
        return replace(document, meta={**document.meta, _ERROR_FIELD: str(readable)}), False

    def _classify(self, client: nace_sdk.NaceClient, document: Document) -> tuple[Document, bool]:
        """
        Sends one document to Drex and adds the answers to a copy of it.

        :param client:
            The synchronous Nace client.
        :param document:
            The document to classify.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises nace_sdk.NaceError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = self._text(document)
        if not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            response = client.system_one(**self._call(text))
        except nace_sdk.NaceError as error:
            return self._fail(document, error)
        return self._succeed(document=document, response=response)

    async def _classify_async(
        self, client: nace_sdk.AsyncNaceClient, document: Document, semaphore: asyncio.Semaphore
    ) -> tuple[Document, bool]:
        """
        Asynchronously sends one document to Drex and adds the answers to a copy of it.

        :param client:
            The asynchronous Nace client.
        :param document:
            The document to classify.
        :param semaphore:
            Limits how many requests run at once.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises nace_sdk.NaceError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = self._text(document)
        if not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            async with semaphore:
                response = await client.system_one(**self._call(text))
        except nace_sdk.NaceError as error:
            return self._fail(document, error)
        return self._succeed(document=document, response=response)

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Answers the questions for each document and adds the answers to its metadata.

        Each document is sent as its own request carrying every question, with up to `max_workers` requests running
        at once.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
              answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises nace_sdk.NaceError:
            If a request fails and `raise_on_failure` is `True`.
        """
        if not documents:
            return {"documents": [], "failed_documents": []}
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        results = map_in_threads(
            lambda document: self._classify(client=client, document=document), documents, self.max_workers
        )
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously answers the questions for each document and adds the answers to its metadata.

        This is the asynchronous version of the `run` method with the same parameters and return values. Up to
        `max_workers` requests run concurrently.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
              answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises nace_sdk.NaceError:
            If a request fails and `raise_on_failure` is `True`.
        """
        if not documents:
            return {"documents": [], "failed_documents": []}
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        # Bound concurrency to max_workers, mirroring the ThreadPoolExecutor in the sync `run`
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [
            asyncio.create_task(self._classify_async(client=self._async_client, document=document, semaphore=semaphore))
            for document in documents
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            # We only expect to get here when `raise_on_failure` is `True`, since `_classify_async` otherwise records a
            # failed request in `failed_documents` instead of raising. Cancel the remaining requests before re-raising.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }
