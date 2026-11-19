# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""NDI component tests against a mocked Nace API: fake keys, *.example hosts and synthetic text only."""

import json

import httpx
import nace_sdk
import pytest

from haystack_integrations.common.nace.client import CLIENT_NAME, _prepare_request, client_options, make_client
from haystack_integrations.components.connectors.nace import NaceWorkspaceSearch
from haystack_integrations.components.converters.nace import NaceDocumentConverter
from haystack_integrations.components.extractors.nace import NaceDocumentExtractor

BASE = "https://console.nace.ai"
WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
FILE_ID = "22222222-2222-2222-2222-222222222222"
JOB_ID = "33333333-3333-3333-3333-333333333333"
SEARCH_JOB_ID = "55555555-5555-5555-5555-555555555555"
UPLOAD_URL = "https://uploads.example/uploads/1"
CONTENT_PATH = f"/v1/documents/jobs/{JOB_ID}/document-content"
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
JOB_ERROR = {"code": "cannot_read_file", "message": "The document service cannot read this file."}
UPLOAD_JOB = {
    "job_id": "44444444-4444-4444-4444-444444444444",
    "kind": "file_upload",
    "status": "succeeded",
    "result": {"file": {"workspace_id": WORKSPACE_ID, "file_id": FILE_ID, "path": "haystack/fake/invoice.html"}},
}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "fake-nace-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


class FakeNdi:
    """Answers a full NDI flow: uploads on the document service's *.example host, everything else at the base URL."""

    def __init__(self, result=None, *, status="succeeded", output=None, result_state=None):
        self.result = result
        self.status = status
        self.output = output
        self.result_state = result_state
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
            return httpx.Response(200, json=self._job(request.url.path))
        if request.url.path == f"/v1/workspaces/{WORKSPACE_ID}/automatic-search":
            job: dict = {"job_id": SEARCH_JOB_ID, "kind": "search", "status": self.status}
            if self.status == "succeeded":
                job["result"] = {"output": self.output}
            else:
                job["error"] = JOB_ERROR
            return httpx.Response(200, json=job)
        if request.url.path == CONTENT_PATH:
            return httpx.Response(200, text=FULL_MARKDOWN, headers={"content-type": "text/markdown"})
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

    # A failed https source is logged by its URL without the query, so a signed query string stays out of the log.
    url = "https://docs.example/invoice.html?sig=secret-signature"
    skipping_url = FakeNdi(status="failed")
    converter = _with_fake_client(NaceDocumentConverter(max_retries=0), skipping_url)
    assert converter.run(sources=[url])["documents"] == []
    assert "https://docs.example/invoice.html" in caplog.text
    assert "secret-signature" not in caplog.text
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
    # The full Markdown is the job's own file: fetched from the base URL origin, with the API key.
    fetch = api.requests[1]
    assert (fetch.method, fetch.url.path) == ("GET", CONTENT_PATH)
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
