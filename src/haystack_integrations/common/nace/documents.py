# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""One Nace document job per source: source selection, upload, waiting and cleanup, shared by the NDI components."""

import mimetypes
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import nace_sdk
from haystack import Document, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

from haystack_integrations.common.nace.concurrency import map_in_threads
from haystack_integrations.common.nace.errors import readable_error

logger = logging.getLogger(__name__)


@contextmanager
def document_job(
    client: nace_sdk.NaceClient,
    operation: Literal["parse", "extract"],
    source: str | Path | ByteStream,
    job_timeout: float,
    **options: Any,
) -> Iterator[nace_sdk.DocumentJob]:
    """
    Runs one NDI document job for one source and always cleans up after it.

    An `https://` string is sent to NDI as a URL source, as given and without downloading it. Every other source is
    read into a `ByteStream` and uploaded into the account's workspace first. The job is created with
    `wait_seconds=60` (NDI holds the create open that long, and zero-retention accounts refuse a shorter wait), then
    waited for until it is terminal. A job that did not succeed raises `nace_sdk.NaceJobFailedError`.

    Use as a context manager: the yielded job's `result` is read inside the `with` block. On exit the job is deleted,
    and an uploaded file is deleted after it, also when the body raises. A cleanup call that fails is logged and
    skipped, so it can neither hide the result nor mask the first error.

    :param client:
        The synchronous Nace client.
    :param operation:
        The NDI operation: `"parse"` or `"extract"`.
    :param source:
        A file path, a `ByteStream`, or an `https://` URL string.
    :param job_timeout:
        How long to wait for the job to finish, in seconds.
    :param options:
        Body fields of the create request, such as `parse_mode` or `schema`.
    :returns:
        A context manager that yields the succeeded `nace_sdk.DocumentJob`.
    :raises nace_sdk.NaceError:
        If the upload, the create or the wait fails, or the job did not succeed.
    :raises OSError:
        If a local source cannot be read.
    :raises ValueError:
        If the file type of a source cannot be told.
    """
    uploaded: nace_sdk.UploadedFile | None = None
    job_id: str | None = None
    try:
        job_source: Any = source
        if not (isinstance(source, str) and source.startswith("https://")):
            bytestream = get_bytestream_from_source(source)
            name = upload_file_name(bytestream)
            uploaded = client.documents.upload(bytestream.data, file_name=name, path=f"haystack/{uuid4().hex}/{name}")
            job_source = uploaded.as_source()
        job = getattr(client.documents, operation)(job_source, wait_seconds=60, **options)
        job_id = job.job_id
        if not job.is_terminal:
            job = client.jobs.wait(job_id, timeout=job_timeout, raise_on_failure=False)
        if job.status != "succeeded":
            raise nace_sdk.NaceJobFailedError(job)
        yield job
    finally:
        if job_id is not None:
            try:
                client.jobs.delete(job_id)
            except nace_sdk.NaceError as error:
                logger.warning("Could not delete NDI job {job_id}: {error}", job_id=job_id, error=error)
        if uploaded is not None:
            try:
                client.documents.delete_file(uploaded.workspace_id, uploaded.file_id)
            except nace_sdk.NaceError as error:
                logger.warning(
                    "Could not delete uploaded NDI file {file_id}: {error}", file_id=uploaded.file_id, error=error
                )


def upload_file_name(bytestream: ByteStream) -> str:
    """
    The file name an uploaded `ByteStream` is given, so NDI can pick its parser from its extension.

    The name is the base name of `meta["file_path"]`, else `meta["file_name"]`, else `"document"`. A name without an
    extension gets the extension of the stream's MIME type; if there is still no extension, the file type cannot be
    told and the source is refused.

    :param bytestream:
        The stream to name.
    :returns:
        The file name, with an extension.
    :raises ValueError:
        If the name has no extension and the MIME type does not name one.
    """
    meta = bytestream.meta
    file_path = str(meta.get("file_path") or "")
    name = os.path.basename(file_path) if file_path else str(meta.get("file_name") or "document")
    if not Path(name).suffix:
        name += mimetypes.guess_extension(bytestream.mime_type or "") or ""
    if not Path(name).suffix:
        msg = (
            f"Cannot tell the file type of {name!r}: the source has no file extension, and its MIME type "
            f"{bytestream.mime_type!r} does not name one. NDI picks its parser from the file extension."
        )
        raise ValueError(msg)
    return name


def job_source(source: str | Path | ByteStream) -> tuple[str | ByteStream, dict[str, Any]]:
    """
    Splits a converter source into what `document_job` is given and the metadata its Documents start from.

    :param source:
        A file path, a `ByteStream`, or an `https://` URL string.
    :returns:
        The source for the job, and its base metadata: the stream's `meta`, or `{"url": source}` for a URL source.
    """
    if isinstance(source, str) and source.startswith("https://"):
        return source, {"url": source}
    bytestream = get_bytestream_from_source(source)
    return bytestream, bytestream.meta


def merged_meta(base_meta: dict[str, Any], meta_for_source: dict[str, Any], store_full_path: bool) -> dict[str, Any]:
    """
    Merges a source's base metadata with the metadata given for it in `run`, like Haystack's own converters.

    :param base_meta:
        The base metadata from `job_source`.
    :param meta_for_source:
        The metadata `run` was given for this source, which wins over the base metadata.
    :param store_full_path:
        Whether to keep `file_path` as it is. If `False`, only its base name is kept.
    :returns:
        The merged metadata.
    """
    meta = {**base_meta, **meta_for_source}
    if not store_full_path and "file_path" in meta:
        file_path = meta.get("file_path")
        if file_path:  # Ensure the value is not None for mypy
            meta["file_path"] = os.path.basename(str(file_path))
    return meta


def convert_sources(
    convert: Callable[[nace_sdk.NaceClient, str | Path | ByteStream, dict[str, Any]], list[Document]],
    client: nace_sdk.NaceClient,
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    max_workers: int,
    raise_on_failure: bool,
) -> list[Document]:
    """
    Converts every source with `convert` on up to `max_workers` threads, keeping the order of `sources`.

    A source that fails (`nace_sdk.NaceError`, `OSError` or `ValueError`) is skipped with a warning when
    `raise_on_failure` is `False`, and its cleanup still runs. With `True`, the first failure raises its readable
    error.

    :param convert:
        The per-source conversion, called with the client, the source and its metadata.
    :param client:
        The synchronous Nace client.
    :param sources:
        The sources, in output order.
    :param meta:
        Metadata for every source, or one dict for all of them, or `None`.
    :param max_workers:
        The most sources converted at once.
    :param raise_on_failure:
        Whether a failed source raises its error instead of being skipped.
    :returns:
        The Documents of all sources that succeeded, in the order of `sources`.
    :raises nace_sdk.NaceError:
        If a source fails and `raise_on_failure` is `True`.
    :raises OSError:
        If a local source cannot be read and `raise_on_failure` is `True`.
    :raises ValueError:
        If the file type of a source cannot be told, or `meta` doesn't match `sources`, and `raise_on_failure` is
        `True`.
    """
    if not sources:
        return []
    pairs = list(zip(sources, normalize_metadata(meta=meta, sources_count=len(sources)), strict=True))
    batches = map_in_threads(
        lambda pair: _convert_one(convert, client, pair[0], pair[1], raise_on_failure), pairs, max_workers
    )
    return [document for batch in batches for document in batch]


def _convert_one(
    convert: Callable[[nace_sdk.NaceClient, str | Path | ByteStream, dict[str, Any]], list[Document]],
    client: nace_sdk.NaceClient,
    source: str | Path | ByteStream,
    meta_for_source: dict[str, Any],
    raise_on_failure: bool,
) -> list[Document]:
    try:
        return convert(client, source, meta_for_source)
    except (nace_sdk.NaceError, OSError, ValueError) as error:
        if not raise_on_failure:
            failure = readable_error(error, "Nace") if isinstance(error, nace_sdk.NaceError) else error
            logger.warning("Converting source {source} failed: {error}. Skipping it.", source=source, error=failure)
            return []
        if isinstance(error, nace_sdk.NaceError):
            raise readable_error(error, "Nace") from error
        raise error
