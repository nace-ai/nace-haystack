# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""Live checks of the NDI components against prod (console.nace.ai). Synthetic data only."""

import os
import uuid

import nace_sdk
import pytest

from haystack_integrations.components.connectors.nace import NaceWorkspaceSearch
from haystack_integrations.components.converters.nace import NaceDocumentConverter
from haystack_integrations.components.extractors.nace import NaceDocumentExtractor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("NACE_API_KEY"), reason="Set NACE_API_KEY to run the live tests."),
]

INVOICE_HTML = b"""<!DOCTYPE html>
<html>
<head><title>Invoice INV-1042</title></head>
<body>
<h1>Invoice INV-1042</h1>
<p>Date: 2026-11-20</p>
<table>
<tr><td>Subtotal</td><td>$261.00</td></tr>
<tr><td>Tax</td><td>$23.00</td></tr>
<tr><td>Total</td><td>$284.00</td></tr>
</table>
</body>
</html>
"""
SCHEMA = {"type": "object", "properties": {"invoice_number": {"type": "string"}}, "required": ["invoice_number"]}
QUERY = "What is the total amount of invoice INV-1042?"


@pytest.fixture
def invoice(tmp_path):
    path = tmp_path / "invoice.html"
    path.write_bytes(INVOICE_HTML)
    return path


def test_converter_live(invoice):
    converter = NaceDocumentConverter(raise_on_failure=True)
    result = converter.run(sources=[invoice])
    assert "INV-1042" in result["documents"][0].content


def test_extractor_live(invoice):
    extractor = NaceDocumentExtractor(schema=SCHEMA, raise_on_failure=True)
    result = extractor.run(sources=[invoice])
    data = result["documents"][0].meta["extraction"]["data"]
    assert data["invoice_number"] == "INV-1042"


def test_workspace_search_live(invoice):
    with nace_sdk.NaceClient() as client:
        name = f"nace-haystack-ndi-tests-{uuid.uuid4().hex[:8]}"
        workspace = client.workspaces.create(name=name)
        workspace_id = str(workspace.workspace_id)
        try:
            uploaded = client.documents.upload(
                invoice.read_bytes(),
                file_name="invoice.html",
                path=f"haystack/{name}/invoice.html",
                workspace_id=workspace_id,
            )
            job = client.workspaces.ingest(workspace_id, file_ids=[uploaded.file_id], wait_seconds=60)
            if not job.is_terminal:
                client.jobs.wait(job.job_id, timeout=300)
            search = NaceWorkspaceSearch(workspace_id=workspace_id)
            answer = search.run(query=QUERY)["answers"][0]
            assert "284" in answer.data
            assert len(answer.documents) >= 1
        finally:
            client.workspaces.delete(workspace_id, confirm_name=name)
