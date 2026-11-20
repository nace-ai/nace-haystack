# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""NDI component tests against a mocked Nace API: fake keys, *.example hosts and synthetic text only."""

import json

import httpx
import nace_sdk
import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.common.nace.client import CLIENT_NAME, _prepare_request, client_options, make_client
from haystack_integrations.common.nace.documents import upload_file_name
from haystack_integrations.components.connectors.nace import NaceWorkspaceSearch
from haystack_integrations.components.converters.nace import NaceDocumentConverter
from haystack_integrations.components.extractors.nace import NaceDocumentExtractor

BASE = "https://console.nace.ai"
WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
FILE_ID = "22222222-2222-2222-2222-222222222222"
JOB_ID = "33333333-3333-3333-3333-333333333333"
SEARCH_JOB_ID = "55555555-5555-5555-5555-555555555555"
UPLOAD_URL = "https://uploads.example/uploads/1"
ARTIFACT_REF = "9f8e7d6c5b4a3210"
CONTENT_PATH = f"/v1/documents/jobs/{JOB_ID}/document-content/{ARTIFACT_REF}"
MARKDOWN = "# Invoice INV-1042\n\nTotal: $284.00"
FULL_MARKDOWN = MARKDOWN + "\n\nThe rest of the full parse result."
ANSWER = "The total of invoice INV-1042 is $284.00."
EVIDENCES = [
    {"source_path": "invoices/invoice.html", "source_file_id": FILE_ID, "page": 3, "quote": "Total: $284.00"},
    {
        "source_path": "invoices/cover.html",
        "source_file_id": FILE_ID,
        "page": None,
        "quote": "Invoice INV-1042",
    },
]
SIGNED_URL = "https://reader:hunter2@docs.example/invoice.html?sig=secret-signature"
UPPERCASE_URL = "HTTPS://reader:hunter3@docs.example/invoice.html?sig=uppercase-secret"
BARE_AUTHORITY = "alice:hunter3@docs.example"
JOB_ERROR = {
    "code": "cannot_read_file",
    "message": f"The document service cannot read this file: {SIGNED_URL}. Retried as {UPPERCASE_URL} and "
    f"{BARE_AUTHORITY}.",
}
UPLOAD_JOB = {
    "job_id": "44444444-4444-4444-4444-444444444444",
    "kind": "file_upload",
    "status": "succeeded",
    "result": {"file": {"workspace_id": WORKSPACE_ID, "file_id": FILE_ID, "path": "haystack/fake/invoice.html"}},
}
KEY_ECHO = {
    "error": {
        "type": "authentication_error",
        "code": "invalid_api_key",
        "message": "Invalid Authorization: Bearer fake-nace-key",
    }
}
JOB_PENDING = {
    "error": {
        "type": "service_unavailable",
        "code": "job_pending",
        "message": "The job's create is not confirmed yet.",
        "retryable": True,
    }
}
CATALOG = {
    "columns": [{"field": "invoice", "kind": "text", "description": "The invoice's number"}],
    "rows": [
        {
            "source_path": "invoices/inv-1042.html",
            "source_file_id": FILE_ID,
            "entry_id": "inv-1042",
            "summary": "Invoice INV-1042, total $284.00",
            "cells": [{"status": "found", "value": "INV-1042", "citation": {}}],
        },
        {
            "source_path": "invoices/inv-1043.html",
            "source_file_id": FILE_ID,
            "entry_id": "inv-1043",
            "summary": "Invoice INV-1043, total $1,204.00",
            "cells": [{"status": "found", "value": "INV-1043", "citation": {}}],
        },
    ],
    "total_matches": 2,
    "total_files": 2,
    "exhaustive": False,
}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "fake-nace-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


class FakeNdi:
    """Answers a full NDI flow: uploads on the document service's *.example host, everything else at the base URL."""

    def __init__(
        self, result=None, *, status="succeeded", output=None, result_state=None, create_error=None, succeed_on_read=1
    ):
        self.result = result
        self.status = status
        self.output = output
        self.result_state = result_state
        self.create_error = create_error
        self.succeed_on_read = succeed_on_read
        self.job_reads = 0
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/documents/upload-grants":
            return httpx.Response(
                200,
                json={
                    "workspace_id": WORKSPACE_ID,
                    "upload_url": UPLOAD_URL,
                    "token": "fake-upload-token",
                    "expires_at": "2026-11-20T12:00:00Z",
                },
            )
        if request.url.host == "uploads.example":
            return httpx.Response(200, json=UPLOAD_JOB)
        if request.url.path in ("/v1/documents/parse", "/v1/documents/extract"):
            if self.create_error is not None:
                return self._create_error_response()
            return httpx.Response(200, json=self._job(request.url.path))
        if request.url.path == f"/v1/workspaces/{WORKSPACE_ID}/automatic-search":
            if self.create_error is not None:
                return self._create_error_response()
            job: dict = {"job_id": SEARCH_JOB_ID, "kind": "search", "status": self.status}
            if self.status == "succeeded":
                job["result"] = {"output": self.output}
            else:
                job["error"] = JOB_ERROR
            return httpx.Response(200, json=job)
        if request.url.path == CONTENT_PATH:
            return httpx.Response(200, text=FULL_MARKDOWN, headers={"content-type": "text/markdown"})
        if request.method == "GET" and request.url.path == f"/v1/documents/jobs/{JOB_ID}":
            # A job read: `running` until the `succeed_on_read`th read answers the finished job.
            self.job_reads += 1
            if self.job_reads >= self.succeed_on_read:
                return httpx.Response(
                    200, json={"job_id": JOB_ID, "kind": "parse", "status": "succeeded", "result": self.result}
                )
            return httpx.Response(200, json={"job_id": JOB_ID, "kind": "parse", "status": "running"})
        if request.method == "DELETE" and request.url.path in (
            f"/v1/documents/jobs/{JOB_ID}",
            f"/v1/documents/jobs/{SEARCH_JOB_ID}",
        ):
            return httpx.Response(200)
        if (
            request.method == "DELETE"
            and request.url.path == f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}"
        ):
            return httpx.Response(
                200,
                json={"object": "file_deletion", "workspace_id": WORKSPACE_ID, "file_id": FILE_ID, "status": "deleted"},
            )
        return httpx.Response(404, json={"error": {"message": f"unexpected {request.method} {request.url}"}})

    def _job(self, path: str) -> dict:
        kind = path.rsplit("/", 1)[-1]
        if self.status != "succeeded":
            return {"job_id": JOB_ID, "kind": kind, "status": self.status, "error": JOB_ERROR}
        job = {"job_id": JOB_ID, "kind": kind, "status": "succeeded", "result": self.result}
        if self.result_state is not None:
            job["result_state"] = self.result_state
        return job

    def _create_error_response(self):
        # A `(status, body)` the gateway answered, or an exception the connection raised.
        if isinstance(self.create_error, Exception):
            raise self.create_error
        return httpx.Response(self.create_error[0], json=self.create_error[1])

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.method == method and request.url.path == path]


def _with_fake_client(component, api: FakeNdi):
    options = client_options(component.api_key, component.api_base_url, component.timeout, component.max_retries) | {
        "transport": httpx.MockTransport(api)
    }
    component._client = make_client(options)
    return component


def _invoice(tmp_path):
    path = tmp_path / "invoice.html"
    path.write_text("<h1>Invoice INV-1042</h1><p>Total: $284.00</p>")
    return path


@pytest.mark.usefixtures("api_key")
def test_converter_local_source_uploads_parses_and_cleans_up(tmp_path):
    api = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 3}, "disclosures": []})
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), api)

    result = converter.run(sources=[_invoice(tmp_path)])

    assert [document.content for document in result["documents"]] == [MARKDOWN]
    assert result["documents"][0].meta == {"file_path": "invoice.html", "page_count": 3}
    # The upload went to the document service's host: its one-use token and a User-Agent, but no API key.
    upload = api.requests[1]
    assert (upload.method, upload.url.host) == ("POST", "uploads.example")
    assert upload.headers["x-upload-token"] == "fake-upload-token"
    assert upload.headers["User-Agent"] == CLIENT_NAME
    assert "authorization" not in upload.headers
    # The parse asked NDI to hold the create open and named the uploaded workspace file as its source.
    parse = api.requests[2]
    assert parse.url.path == "/v1/documents/parse"
    assert parse.url.params["wait_seconds"] == "60"
    assert parse.headers["authorization"] == "Bearer fake-nace-key"
    assert json.loads(parse.content)["source"] == {
        "type": "workspace_file",
        "workspace_id": WORKSPACE_ID,
        "file_id": FILE_ID,
    }
    # The job and the uploaded file were deleted after the run.
    assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1
    assert len(api.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1
    # A stream named only by its MIME type gets its extension from it, with the MIME type's parameters ignored.
    assert upload_file_name(ByteStream(data=b"x", mime_type="text/html; charset=utf-8")).endswith((".html", ".htm"))


@pytest.mark.usefixtures("api_key")
def test_converter_https_url_source_is_sent_as_given():
    url = "https://docs.example/invoices/inv-1042.html"
    api = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1}, "disclosures": []})
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), api)

    result = converter.run(sources=[url])

    assert [document.content for document in result["documents"]] == [MARKDOWN]
    assert result["documents"][0].meta == {"url": url, "page_count": 1}
    source = json.loads(api.requests[0].content)["source"]
    assert source["type"] == "url"
    assert source["url"] == url
    # No upload, so no file to delete: only the create and the job delete were sent.
    assert [request.method for request in api.requests] == ["POST", "DELETE"]
    assert api.requests[1].url.path == f"/v1/documents/jobs/{JOB_ID}"


@pytest.mark.usefixtures("api_key")
def test_converter_page_chunking_returns_one_document_per_chunk():
    api = FakeNdi(
        {
            "document": {
                "markdown": MARKDOWN,
                "page_count": 2,
                "chunks": [
                    {"id": "chunk-1", "content": "# Invoice INV-1042", "location": {"page": 1}},
                    {"id": "chunk-2", "content": "Total: $284.00", "location": {"page": 2}},
                ],
            },
            "disclosures": [],
        }
    )
    converter = _with_fake_client(NaceDocumentConverter(chunking="page", max_retries=0), api)

    result = converter.run(sources=["https://docs.example/invoice.html"])

    assert json.loads(api.requests[0].content)["chunking"] == {"strategy": "page"}
    assert [document.content for document in result["documents"]] == ["# Invoice INV-1042", "Total: $284.00"]
    assert [document.meta["page_number"] for document in result["documents"]] == [1, 2]
    assert result["documents"][0].meta == {
        "url": "https://docs.example/invoice.html",
        "page_count": 2,
        "chunk_id": "chunk-1",
        "page_number": 1,
    }


@pytest.mark.usefixtures("api_key")
def test_converter_failed_job_skips_or_raises_and_still_cleans_up(tmp_path, caplog):
    source = _invoice(tmp_path)

    skipping = FakeNdi(status="failed")
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), skipping)
    assert converter.run(sources=[source])["documents"] == []
    assert len(skipping.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1
    assert len(skipping.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1

    raising = FakeNdi(status="failed")
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), raising)
    with pytest.raises(nace_sdk.NaceJobFailedError, match="cannot read this file"):
        converter.run(sources=[source])
    assert len(raising.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1
    assert len(raising.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1

    # Raw bytes, a source type Haystack converters refuse, and a malformed https source are skipped with warnings,
    # and the valid source still converts.
    mixed = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1}, "disclosures": []})
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), mixed)
    result = converter.run(
        sources=[b"<h1>raw bytes</h1>", "https://[broken/invoice.pdf", "https://docs.example/invoice.html"]
    )
    assert [document.content for document in result["documents"]] == [MARKDOWN]
    assert "Converting source bytes failed" in caplog.text
    assert "Converting source https URL failed" in caplog.text

    # A failed signed https source is logged without its user info or its query, also when the error quotes it
    # signed, upper-case, or bare without a scheme.
    skipping_url = FakeNdi(status="failed")
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), skipping_url)
    assert converter.run(sources=[SIGNED_URL])["documents"] == []
    assert "https://docs.example/invoice.html" in caplog.text
    assert "hunter2" not in caplog.text
    assert "hunter3" not in caplog.text
    assert "secret-signature" not in caplog.text
    assert "uppercase-secret" not in caplog.text
    assert len(skipping_url.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_converter_succeeded_job_without_result_raises_and_still_cleans_up():
    api = FakeNdi(None, result_state="not_retained")
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), api)

    with pytest.raises(nace_sdk.NaceError, match=rf"{JOB_ID}.*not_retained"):
        converter.run(sources=["https://docs.example/invoice.html"])

    assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_converter_truncated_result_fetches_the_full_markdown():
    api = FakeNdi(
        {
            "document": {
                "markdown": "# Invoice INV-10",
                "page_count": 3,
                "content_url": CONTENT_PATH,
            },
            "disclosures": [{"code": "content_truncated", "target": "document_content", "message": "over 1 MiB"}],
        }
    )
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), api)

    result = converter.run(sources=["https://docs.example/invoice.html"])

    assert result["documents"][0].content == FULL_MARKDOWN
    # The full Markdown is the job's own file, at its artifact reference: fetched from the base URL origin with the
    # API key.
    fetch = api.requests[1]
    assert fetch.method == "GET"
    assert fetch.url.path == CONTENT_PATH
    assert fetch.url.path.endswith(f"/document-content/{ARTIFACT_REF}")
    assert fetch.url.host == "console.nace.ai"
    assert fetch.headers["authorization"] == "Bearer fake-nace-key"
    assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_extractor_returns_data_and_fields_and_cleans_up(tmp_path):
    fields = [
        {
            "path": "invoice_number",
            "value": "INV-1042",
            "status": "found",
            "confidence": 0.98,
            "citations": [{"quote": "Invoice INV-1042", "page": 1}],
        }
    ]
    api = FakeNdi({"data": {"invoice_number": "INV-1042"}, "fields": fields})
    schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}, "required": ["invoice_number"]}
    extractor = _with_fake_client(NaceDocumentExtractor(schema=schema, max_retries=0), api)

    result = extractor.run(sources=[_invoice(tmp_path)])

    documents = result["documents"]
    assert json.loads(documents[0].content) == {"invoice_number": "INV-1042"}
    assert documents[0].meta == {
        "file_path": "invoice.html",
        "extraction": {"data": {"invoice_number": "INV-1042"}, "fields": fields},
    }
    extract = api.requests[2]
    assert extract.url.path == "/v1/documents/extract"
    assert extract.url.params["wait_seconds"] == "60"
    body = json.loads(extract.content)
    assert body["schema"] == schema
    assert body["parse_mode"] == "low"
    assert body["source"] == {"type": "workspace_file", "workspace_id": WORKSPACE_ID, "file_id": FILE_ID}
    assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1
    assert len(api.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1


@pytest.mark.parametrize(
    "output",
    [
        {"answer": ANSWER, "confidence": "certain", "evidences": EVIDENCES},
        {
            "result_type": "automatic_agent",
            "answer": {
                "answer": ANSWER,
                "confidence": "certain",
                "evidences": EVIDENCES,
                "structured_answer": {"total": "$284.00"},
            },
        },
    ],
    ids=["fact", "agent"],
)
@pytest.mark.usefixtures("api_key")
def test_search_returns_one_answer_with_unscored_evidence_and_deletes_the_job(output):
    api = FakeNdi(output=output)
    search = _with_fake_client(NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), api)

    result = search.run(query="What is the total of invoice INV-1042?")

    create = api.requests[0]
    assert create.url.path == f"/v1/workspaces/{WORKSPACE_ID}/automatic-search"
    assert create.url.params["wait_seconds"] == "60"
    assert json.loads(create.content) == {"query": "What is the total of invoice INV-1042?", "mode": "auto"}
    (answer,) = result["answers"]
    assert answer.data == ANSWER
    assert answer.query == "What is the total of invoice INV-1042?"
    assert answer.meta == {"confidence": "certain"}
    assert [document.content for document in answer.documents] == ["Total: $284.00", "Invoice INV-1042"]
    assert answer.documents[0].meta == {
        "source_path": "invoices/invoice.html",
        "source_file_id": FILE_ID,
        "page_number": 3,
    }
    assert answer.documents[1].meta == {"source_path": "invoices/cover.html", "source_file_id": FILE_ID}
    assert all(document.score is None for document in answer.documents)
    assert [request.method for request in api.requests] == ["POST", "DELETE"]
    assert api.requests[1].url.path == f"/v1/documents/jobs/{SEARCH_JOB_ID}"


@pytest.mark.usefixtures("api_key")
def test_request_hook_refuses_credentials_and_plain_http_at_other_origins():
    with pytest.raises(ValueError, match="different API origin"):
        _prepare_request(
            httpx.Request("GET", "https://other.example/file", headers={"Authorization": "Bearer key"}), BASE
        )
    with pytest.raises(ValueError, match="different API origin"):
        _prepare_request(httpx.Request("GET", "http://uploads.example/file"), BASE)
    # A request to another origin never carries the Vercel bypass secret, not even to another Nace origin.
    to_nace = httpx.Request(
        "GET", "https://drex.nace.ai/file", headers={"x-vercel-protection-bypass": "fake-bypass-secret"}
    )
    _prepare_request(to_nace, BASE)
    assert "x-vercel-protection-bypass" not in to_nace.headers


@pytest.mark.usefixtures("api_key")
def test_an_error_that_echoes_the_api_key_never_leaks_it(tmp_path, caplog):
    # A gateway can answer a failed request by echoing the Authorization value it got: neither the
    # skip warning nor a raised error may carry the key.
    skipping = FakeNdi(create_error=(401, KEY_ECHO))
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), skipping)
    assert converter.run(sources=[_invoice(tmp_path)])["documents"] == []
    assert "fake-nace-key" not in caplog.text
    assert "[redacted]" in caplog.text
    assert len(skipping.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1

    raising = FakeNdi(create_error=(401, KEY_ECHO))
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), raising)
    with pytest.raises(nace_sdk.NaceAPIError) as raised:
        converter.run(sources=[_invoice(tmp_path)])
    assert isinstance(raised.value, nace_sdk.NaceAPIError)
    assert "fake-nace-key" not in str(raised.value)
    assert "fake-nace-key" not in json.dumps(raised.value.body)
    assert "[redacted]" in str(raised.value)

    search = _with_fake_client(
        NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), FakeNdi(create_error=(401, KEY_ECHO))
    )
    with pytest.raises(nace_sdk.NaceAPIError) as raised:
        search.run(query="What is the total of invoice INV-1042?")
    assert "fake-nace-key" not in str(raised.value)
    assert "fake-nace-key" not in json.dumps(raised.value.body)


@pytest.mark.usefixtures("api_key")
def test_a_presigned_url_source_keeps_its_signature_out_of_document_meta():
    api = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1}, "disclosures": []})
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), api)

    result = converter.run(sources=[SIGNED_URL])

    # NDI gets the signed URL as given: it needs the signature to download the file. The Document's
    # metadata keeps only the scheme, host and path.
    assert json.loads(api.requests[0].content)["source"]["url"] == SIGNED_URL
    document = result["documents"][0]
    assert document.meta["url"] == "https://docs.example/invoice.html"
    serialized = json.dumps(document.to_dict())
    assert "secret-signature" not in serialized
    assert "hunter2" not in serialized


@pytest.mark.usefixtures("api_key")
def test_agent_search_catalog_answer_keeps_its_rows_and_the_exhausted_flag():
    output = {
        "result_type": "automatic_agent",
        "answer": {
            "answer": None,
            "confidence": "medium",
            "evidences": [],
            "exhausted": True,
            "grounding_job_ids": None,
        },
        "catalog": CATALOG,
    }
    api = FakeNdi(output=output)
    search = _with_fake_client(NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), api)

    result = search.run(query="Which invoices are over $10,000?")

    (answer,) = result["answers"]
    assert answer.data == ""
    assert answer.meta["confidence"] == "medium"
    assert answer.meta["exhausted"] is True
    assert answer.meta["catalog"] == CATALOG
    assert [row["entry_id"] for row in answer.meta["catalog"]["rows"]] == ["inv-1042", "inv-1043"]
    assert answer.documents == []
    assert len(api.sent("DELETE", f"/v1/documents/jobs/{SEARCH_JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_a_create_that_ends_job_pending_keeps_the_key_and_cleans_up_what_is_known(tmp_path):
    # NDI answers 503 job_pending when a create is not confirmed; the job may still run. Its id is
    # not known, so the raised error must name the create's idempotency key to resume and delete it,
    # and the uploaded file must still be deleted.
    api = FakeNdi(create_error=(503, JOB_PENDING))
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), api)

    with pytest.raises(nace_sdk.NaceError) as raised:
        converter.run(sources=[_invoice(tmp_path)])

    error = raised.value
    key = api.sent("POST", "/v1/documents/parse")[0].headers["idempotency-key"]
    assert error.idempotency_key == key
    assert key in str(error)
    assert "job_pending" in str(error)
    assert len(api.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1
    assert api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}") == []

    search_api = FakeNdi(create_error=(503, JOB_PENDING))
    search = _with_fake_client(NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), search_api)
    with pytest.raises(nace_sdk.NaceError) as raised:
        search.run(query="Which invoices are over $10,000?")
    search_key = search_api.sent("POST", f"/v1/workspaces/{WORKSPACE_ID}/automatic-search")[0].headers[
        "idempotency-key"
    ]
    assert raised.value.idempotency_key == search_key
    assert search_key in str(raised.value)
    assert "job_pending" in str(raised.value)


@pytest.mark.usefixtures("api_key")
def test_malformed_successful_results_raise_instead_of_losing_the_source():
    # A succeeded job whose result misses the fields the component maps produced an empty Document,
    # an extraction Document of `null`, or a dropped source: all three raise instead.
    empty = FakeNdi({})
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), empty)
    with pytest.raises(nace_sdk.NaceError, match="without document Markdown"):
        converter.run(sources=["https://docs.example/invoice.html"])
    assert len(empty.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1

    chunkless = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1, "chunks": []}, "disclosures": []})
    converter = _with_fake_client(
        NaceDocumentConverter(chunking="page", max_retries=0, raise_on_failure=True), chunkless
    )
    with pytest.raises(nace_sdk.NaceError, match="without chunks"):
        converter.run(sources=["https://docs.example/invoice.html"])
    assert len(chunkless.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1

    dataless = FakeNdi({})
    extractor = _with_fake_client(
        NaceDocumentExtractor(schema={"type": "object"}, max_retries=0, raise_on_failure=True), dataless
    )
    with pytest.raises(nace_sdk.NaceError, match="without data"):
        extractor.run(sources=["https://docs.example/invoice.html"])
    assert len(dataless.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
@pytest.mark.parametrize("job_timeout", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_non_finite_job_timeouts_are_refused(job_timeout):
    # An infinite or undefined timeout would make a job that stays running hang the pipeline.
    with pytest.raises(ValueError, match="finite"):
        NaceDocumentConverter(job_timeout=job_timeout)
    with pytest.raises(ValueError, match="finite"):
        NaceDocumentExtractor(schema={"type": "object"}, job_timeout=job_timeout)
    with pytest.raises(ValueError, match="finite"):
        NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, job_timeout=job_timeout)


@pytest.mark.usefixtures("api_key")
def test_the_create_hold_and_the_polls_never_outlast_the_job_timeout():
    # A job that is still running when the create returns is polled. The create's hold and every
    # read's hold and HTTP timeout stay within the job budget, so no single request can outlast
    # `job_timeout` while the job waits.
    parse_result = {"document": {"markdown": MARKDOWN, "page_count": 1}, "disclosures": []}
    default = FakeNdi(parse_result, status="running", succeed_on_read=1)
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), default)
    result = converter.run(sources=["https://docs.example/invoice.html"])
    assert [document.content for document in result["documents"]] == [MARKDOWN]
    # With the usual 600 s budget the create holds the full 60 s and each read the full 25 s.
    assert default.requests[0].url.params["wait_seconds"] == "60"
    read = default.sent("GET", f"/v1/documents/jobs/{JOB_ID}")[0]
    assert read.url.params["wait_seconds"] == "25"

    short = FakeNdi(parse_result, status="running", succeed_on_read=2)
    converter = _with_fake_client(NaceDocumentConverter(job_timeout=45.0, max_retries=0), short)
    result = converter.run(sources=["https://docs.example/invoice.html"])
    assert [document.content for document in result["documents"]] == [MARKDOWN]
    # With 45 s left the create asks for a shorter hold than the budget, and the reads are sent
    # without a hold: a held read's HTTP timeout cannot drop below nace-sdk's 95 s floor.
    assert int(short.requests[0].url.params["wait_seconds"]) <= 45
    reads = short.sent("GET", f"/v1/documents/jobs/{JOB_ID}")
    assert len(reads) == 2
    assert all("wait_seconds" not in read.url.params for read in reads)
    assert len(short.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_null_or_contentless_chunks_raise_or_skip_with_a_warning(caplog):
    # A null chunk crashed the run with an AttributeError, and a chunk without content became an
    # empty Document: both are refused as a malformed result instead.
    for chunks in ([None], [{"id": "chunk-1", "content": None}], [{"id": "chunk-1"}]):
        api = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1, "chunks": chunks}, "disclosures": []})
        converter = _with_fake_client(NaceDocumentConverter(chunking="page", max_retries=0, raise_on_failure=True), api)
        with pytest.raises(nace_sdk.NaceError, match="chunk 1 of 1"):
            converter.run(sources=["https://docs.example/invoice.html"])
        assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1

    skipping = FakeNdi({"document": {"markdown": MARKDOWN, "page_count": 1, "chunks": [None]}, "disclosures": []})
    converter = _with_fake_client(NaceDocumentConverter(chunking="page", max_retries=0), skipping)
    assert converter.run(sources=["https://docs.example/invoice.html"])["documents"] == []
    assert "chunk 1 of 1" in caplog.text


@pytest.mark.usefixtures("api_key")
def test_a_succeeded_search_without_output_raises_and_still_deletes_the_job():
    api = FakeNdi(output=None)
    search = _with_fake_client(NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), api)

    with pytest.raises(nace_sdk.NaceError, match="without an answer output"):
        search.run(query="What is the total of invoice INV-1042?")

    assert len(api.sent("DELETE", f"/v1/documents/jobs/{SEARCH_JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_a_lost_create_response_cleans_up_what_is_known(tmp_path):
    # A create whose response is lost (a timeout after NDI accepted it) can leave a job running;
    # its id is not known, so no job is deleted. The uploaded file is still deleted.
    api = FakeNdi(create_error=httpx.ReadTimeout("the create timed out"))
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0, raise_on_failure=True), api)

    with pytest.raises(nace_sdk.NaceError, match="Request timed out"):
        converter.run(sources=[_invoice(tmp_path)])

    assert len(api.sent("DELETE", f"/v1/documents/workspaces/{WORKSPACE_ID}/files/{FILE_ID}")) == 1
    assert api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}") == []


@pytest.mark.usefixtures("api_key")
def test_a_chunk_with_a_non_object_location_keeps_its_content():
    # A location that is not an object (a string, say) must not crash the run: it is ignored, and
    # the chunk's content is kept.
    api = FakeNdi(
        {
            "document": {
                "markdown": MARKDOWN,
                "page_count": 2,
                "chunks": [
                    {"id": "chunk-1", "content": "# Invoice INV-1042", "location": "page 1 of 2"},
                    {"id": "chunk-2", "content": "Total: $284.00", "location": None},
                ],
            },
            "disclosures": [],
        }
    )
    converter = _with_fake_client(NaceDocumentConverter(chunking="page", max_retries=0), api)

    result = converter.run(sources=["https://docs.example/invoice.html"])

    assert [document.content for document in result["documents"]] == ["# Invoice INV-1042", "Total: $284.00"]
    assert result["documents"][0].meta["chunk_id"] == "chunk-1"
    assert "page_number" not in result["documents"][0].meta
    assert "page_number" not in result["documents"][1].meta
    assert len(api.sent("DELETE", f"/v1/documents/jobs/{JOB_ID}")) == 1


@pytest.mark.usefixtures("api_key")
def test_an_agent_answer_without_text_or_catalog_raises_and_still_deletes_the_job():
    # An agent answer that holds neither answer text nor a catalog table is a malformed result: it
    # raises like the others, instead of returning an empty GeneratedAnswer.
    output = {"result_type": "automatic_agent", "answer": {}}
    api = FakeNdi(output=output)
    search = _with_fake_client(NaceWorkspaceSearch(workspace_id=WORKSPACE_ID, max_retries=0), api)

    with pytest.raises(nace_sdk.NaceError, match="no answer text and no catalog"):
        search.run(query="Which invoices are over $10,000?")

    assert len(api.sent("DELETE", f"/v1/documents/jobs/{SEARCH_JOB_ID}")) == 1
