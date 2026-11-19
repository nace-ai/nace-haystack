# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, Literal

import nace_sdk
from haystack import Document, component, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.nace.client import bind_credentials, client_options, make_client
from haystack_integrations.common.nace.documents import convert_sources, document_job, job_source, merged_meta

logger = logging.getLogger(__name__)


@component
class NaceDocumentExtractor:
    """
    Fills a JSON Schema from each file with Nace Document Intelligence (NDI) Extract, as one Document per source.

    A source is a file path, a `ByteStream`, or an `https://` URL string, which NDI downloads itself. An `https://`
    source must end in a file name with an extension (for example `.pdf`), because NDI picks its parser from it; a
    URL without one is refused. Files are uploaded into your Nace workspace for the extraction and deleted again
    after it, together with the extract job; a URL source is sent as given. NDI runs in the Nace cloud, so the
    source is sent there.

    Each Document's `content` is the extracted `data` as JSON, and `meta["extraction"]` keeps that `data` together
    with NDI's per-field report: each field's `value`, `status` and `confidence`. Sources run on up to `max_workers`
    threads, in the order of `sources`. A source that fails is skipped with a warning, or raises its readable error
    when `raise_on_failure` is `True`.

    ### Usage example

    ```python
    from haystack_integrations.components.extractors.nace import NaceDocumentExtractor

    extractor = NaceDocumentExtractor(
        schema={
            "type": "object",
            "properties": {"invoice_number": {"type": "string"}},
            "required": ["invoice_number"],
        }
    )
    result = extractor.run(sources=["invoice.html"])
    print(result["documents"][0].meta["extraction"]["data"])
    # {'invoice_number': 'INV-1042'}
    ```
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        instructions: str | None = None,
        parse_mode: Literal["low", "medium", "high"] = "low",
        api_key: Secret | None = None,
        api_base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        job_timeout: float = 600.0,
        max_workers: int = 3,
        raise_on_failure: bool = False,
        store_full_path: bool = False,
    ) -> None:
        """
        Creates a NaceDocumentExtractor.

        :param schema:
            The JSON Schema of the fields to extract, as a dict. The NDI API validates its contents.
        :param instructions:
            Optional extra instructions NDI reads with the schema.
        :param parse_mode:
            How much of the file NDI reads. `"low"` is the cheapest and the default.
        :param api_key:
            A provider key. Defaults to `NACE_API_KEY` on Nace production origins; required for custom hosts.
        :param api_base_url:
            Base URL of the API. If `None`, the `NACE_BASE_URL` environment variable is used, or
            `https://console.nace.ai` if it is unset.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, 60 seconds are used.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param job_timeout:
            How long to wait for each NDI job to finish, in seconds.
        :param max_workers:
            Maximum number of sources extracted at once. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed source raises its error. If `False`, a failed source is skipped with a
            warning.
        :param store_full_path:
            If `False`, only the base name of a local file is stored in the Document metadata.
        :raises ValueError:
            If `schema` is not a non-empty dict, `max_workers` is smaller than 1 or `job_timeout` is not greater
            than 0.
        """
        if not isinstance(schema, dict) or not schema:
            msg = "`schema` must be a non-empty dict holding the JSON Schema of the fields to extract."
            raise ValueError(msg)
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)
        if job_timeout <= 0:
            msg = f"`job_timeout` must be greater than 0, got {job_timeout}."
            raise ValueError(msg)

        self.schema = schema
        self.instructions = instructions
        self.parse_mode = parse_mode
        self.api_key, self.api_base_url = bind_credentials(api_key, api_base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.job_timeout = job_timeout
        self.max_workers = max_workers
        self.raise_on_failure = raise_on_failure
        self.store_full_path = store_full_path
        self._client: nace_sdk.NaceClient | None = None

    def _client_options(self) -> dict[str, Any]:
        return client_options(self.api_key, self.api_base_url, self.timeout, self.max_retries)

    def warm_up(self) -> None:
        """
        Creates the synchronous Nace client.
        """
        if self._client is None:
            self._client = make_client(self._client_options())

    def close(self) -> None:
        """
        Releases the synchronous Nace client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            schema=self.schema,
            instructions=self.instructions,
            parse_mode=self.parse_mode,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            job_timeout=self.job_timeout,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
            store_full_path=self.store_full_path,
        )

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"schema": self.schema, "parse_mode": self.parse_mode}
        if self.instructions is not None:
            options["instructions"] = self.instructions
        return options

    def _extract_source(
        self, client: nace_sdk.NaceClient, source: str | Path | ByteStream, meta_for_source: dict[str, Any]
    ) -> list[Document]:
        source_for_job, base_meta = job_source(source)
        with document_job(client, "extract", source_for_job, self.job_timeout, **self._options()) as job:
            result = job.result or {}
            meta = merged_meta(base_meta, meta_for_source, self.store_full_path)
            meta["extraction"] = {"data": result.get("data"), "fields": result.get("fields")}
            content = json.dumps(result.get("data"), ensure_ascii=False)
            return [Document(content=content, meta=meta)]

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Fills the schema from each source with NDI Extract, as one Document per source.

        Each source is its own NDI job, with up to `max_workers` jobs running at once. The uploaded file and the
        job of each source are deleted after its extraction.

        :param sources:
            File paths, `ByteStream` objects, or `https://` URL strings.
        :param meta:
            Optional metadata to attach to the Documents. This value can be either a single dictionary or a list of
            dictionaries with the same length as `sources`.
        :returns:
            A dictionary with the following keys:
            - `documents`: One Document per source, in the order of `sources`.
        :raises nace_sdk.NaceError:
            If a source fails and `raise_on_failure` is `True`.
        :raises OSError:
            If a local source cannot be read and `raise_on_failure` is `True`.
        :raises ValueError:
            If the file type of a source cannot be told, or `meta` doesn't match `sources`, and `raise_on_failure`
            is `True`.
        """
        if not sources:
            return {"documents": []}
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        documents = convert_sources(
            self._extract_source,
            self._client,
            sources,
            meta,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )
        return {"documents": documents}
