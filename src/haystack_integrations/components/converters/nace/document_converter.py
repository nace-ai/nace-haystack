# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Any, Literal

import nace_sdk
from haystack import Document, component, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.nace.client import bind_credentials, client_options, make_client
from haystack_integrations.common.nace.documents import convert_sources, document_job, job_source, merged_meta

logger = logging.getLogger(__name__)

TRUNCATED_RESULT_MESSAGE = (
    "The NDI result is larger than 1 MiB and was cut. A cut preview is never returned as the full content."
)


@component
class NaceDocumentConverter:
    """
    Converts files into Markdown Documents with Nace Document Intelligence (NDI) Parse.

    A source is a file path, a `ByteStream`, or an `https://` URL string, which NDI downloads itself. An `https://`
    source must end in a file name with an extension (for example `.pdf`), because NDI picks its parser from it; a
    URL without one is refused. Files are uploaded into your Nace workspace for the parse and deleted again after
    it, together with the parse job; a URL source is sent as given. NDI runs in the Nace cloud, so the source is
    sent there.

    With `chunking=None` (the default) each source becomes one Document whose content is the full Markdown NDI
    produced. With `chunking="page"` or `chunking="section"`, each chunk becomes its own Document, with `chunk_id`
    and, for page chunks, `page_number` in its metadata. A result NDI cut at 1 MiB is never returned as the full
    content: the full Markdown of an unchunked parse is fetched instead, and every other cut result raises.

    Sources run on up to `max_workers` threads, in the order of `sources`. A source that fails is skipped with a
    warning, or raises its readable error when `raise_on_failure` is `True`.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.nace import NaceDocumentConverter

    converter = NaceDocumentConverter()
    result = converter.run(sources=["invoice.html"])
    print(result["documents"][0].content)
    # '# Invoice INV-1042 ...'
    ```
    """

    def __init__(
        self,
        *,
        parse_mode: Literal["low", "medium", "high"] = "low",
        chunking: Literal["page", "section"] | None = None,
        table_format: Literal["html", "markdown"] = "html",
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
        Creates a NaceDocumentConverter.

        :param parse_mode:
            How much of the file NDI reads. `"low"` is the cheapest and the default.
        :param chunking:
            `None` for one Document per source with the full Markdown, `"page"` or `"section"` for one Document per
            chunk.
        :param table_format:
            How NDI renders tables in the Markdown.
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
            Maximum number of sources parsed at once. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed source raises its error. If `False`, a failed source is skipped with a
            warning.
        :param store_full_path:
            If `False`, only the base name of a local file is stored in the Document metadata.
        :raises ValueError:
            If `max_workers` is smaller than 1 or `job_timeout` is not greater than 0.
        """
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)
        if job_timeout <= 0:
            msg = f"`job_timeout` must be greater than 0, got {job_timeout}."
            raise ValueError(msg)

        self.parse_mode = parse_mode
        self.chunking = chunking
        self.table_format = table_format
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
            parse_mode=self.parse_mode,
            chunking=self.chunking,
            table_format=self.table_format,
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
        options: dict[str, Any] = {
            "parse_mode": self.parse_mode,
            "output": {"formats": ["markdown"], "table_format": self.table_format},
        }
        if self.chunking is not None:
            options["chunking"] = {"strategy": self.chunking}
        return options

    def _parse_source(
        self, client: nace_sdk.NaceClient, source: str | Path | ByteStream, meta_for_source: dict[str, Any]
    ) -> list[Document]:
        source_for_job, base_meta = job_source(source)
        with document_job(client, "parse", source_for_job, self.job_timeout, **self._options()) as job:
            result = job.result or {}
            document = result.get("document") or {}
            meta = merged_meta(base_meta, meta_for_source, self.store_full_path)
            if document.get("page_count") is not None:
                meta["page_count"] = document["page_count"]
            truncations = _truncations(result)
            if self.chunking is None:
                if truncations:
                    return [Document(content=self._full_markdown(client, job, document, truncations), meta=meta)]
                return [Document(content=document.get("markdown") or "", meta=meta)]
            if truncations:
                raise nace_sdk.NaceError(TRUNCATED_RESULT_MESSAGE)
            documents = []
            for chunk in document.get("chunks") or []:
                chunk_meta = {**meta, "chunk_id": chunk.get("id")}
                page = (chunk.get("location") or {}).get("page")
                if isinstance(page, int) and not isinstance(page, bool):
                    chunk_meta["page_number"] = page
                documents.append(Document(content=chunk.get("content") or "", meta=chunk_meta))
            return documents

    def _full_markdown(
        self,
        client: nace_sdk.NaceClient,
        job: nace_sdk.DocumentJob,
        document: dict[str, Any],
        truncations: list[dict[str, Any]],
    ) -> str:
        """
        Fetches the full Markdown of a truncated unchunked parse from its `content_url`.

        :param client:
            The synchronous Nace client.
        :param job:
            The succeeded parse job, whose file the Markdown is.
        :param document:
            The parse result's `document`, with the truncated preview and the `content_url`.
        :param truncations:
            The result's `content_truncated` disclosures.
        :returns:
            The full Markdown.
        :raises nace_sdk.NaceError:
            If the truncation is not one the full Markdown can be fetched for.
        """
        content_url = document.get("content_url")
        if content_url and any(item.get("target") == "document_content" for item in truncations):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                target = Path(folder) / "content.md"
                client.jobs.download(job.job_id, content_url, target)
                return target.read_text(encoding="utf-8")
        raise nace_sdk.NaceError(TRUNCATED_RESULT_MESSAGE)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Converts each source into one Document, or one per chunk, with NDI Parse.

        Each source is its own NDI job, with up to `max_workers` jobs running at once. The uploaded file and the
        job of each source are deleted after its parse.

        :param sources:
            File paths, `ByteStream` objects, or `https://` URL strings.
        :param meta:
            Optional metadata to attach to the Documents. This value can be either a single dictionary or a list of
            dictionaries with the same length as `sources`.
        :returns:
            A dictionary with the following keys:
            - `documents`: The created Documents, in the order of `sources`.
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
            self._parse_source,
            self._client,
            sources,
            meta,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )
        return {"documents": documents}


def _truncations(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in result.get("disclosures") or []
        if isinstance(item, dict) and item.get("code") == "content_truncated"
    ]
