# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math

import httpx
import nace_sdk
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.common.nace.client import make_async_client, make_client
from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier
from haystack_integrations.components.evaluators.nace import NaceContextRelevanceEvaluator


def component_for(kind, raises):
    options = {"api_key": Secret.from_token("fake-key"), "max_workers": 1, "max_retries": 0, "raise_on_failure": raises}
    if kind == "classifier":
        return NaceDocumentClassifier(questions={"verdict": {"type": "noul", "instructions": "ok?"}}, **options)
    return NaceContextRelevanceEvaluator(**options)


def inputs_for(kind, count):
    if kind == "classifier":
        return {"documents": [Document(content=str(i)) for i in range(count)]}
    return {"questions": [str(i) for i in range(count)], "contexts": [["text"]] * count}


@pytest.mark.parametrize("kind", ["classifier", "evaluator"])
@pytest.mark.parametrize("yielding", [False, True])
@pytest.mark.parametrize("raises", [False, True])
async def test_async_failure_stops_unsent_work_only_when_requested(kind, yielding, raises):
    seen = []

    async def handler(request):
        seen.append(request)
        if yielding:
            await asyncio.sleep(0)
        return httpx.Response(422, json={"error": {"message": "invalid"}})

    component = component_for(kind, raises)
    component._async_client = make_async_client(
        component._client_options() | {"transport": httpx.MockTransport(handler)}
    )
    try:
        if raises:
            with pytest.raises(nace_sdk.NaceUnprocessableEntityError):
                await component.run_async(**inputs_for(kind, 10))
        else:
            result = await component.run_async(**inputs_for(kind, 10))
            assert len(result["failed_documents"] if kind == "classifier" else result["results"]) == 10
        assert len(seen) == (1 if raises else 10)
    finally:
        await component.close_async()


@pytest.mark.parametrize("kind", ["classifier", "evaluator"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("answer", [None, {"type": "noul", "noul": None}, {"type": "noul", "noul": 2}])
async def test_malformed_success_obeys_batch_failure_policy(kind, asynchronous, raises, answer):
    body = {"answers": {} if answer is None else {"verdict": answer}}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
    component = component_for(kind, raises)
    options = component._client_options() | {"transport": transport}
    if asynchronous:
        component._async_client = make_async_client(options)
    else:
        component._client = make_client(options)

    async def run():
        if asynchronous:
            return await component.run_async(**inputs_for(kind, 1))
        return component.run(**inputs_for(kind, 1))

    try:
        if raises:
            with pytest.raises(nace_sdk.NaceAPIResponseValidationError):
                await run()
        else:
            result = await run()
            if kind == "classifier":
                assert len(result["failed_documents"]) == 1
                assert not result["documents"]
            else:
                assert math.isnan(result["score"])
                assert result["results"][0]["status"] == "error"
    finally:
        component.close()
        await component.close_async()
