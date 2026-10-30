# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from haystack import Pipeline
from haystack.core.errors import DeserializationError
from haystack.utils import Secret

from haystack_integrations.common.nace.client import client_options, make_async_client, make_client
from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier
from haystack_integrations.components.evaluators.nace import NaceContextRelevanceEvaluator, NaceFaithfulnessEvaluator
from haystack_integrations.components.routers.nace import NaceTextRouter

FACTORIES = [
    lambda **kwargs: NaceTextRouter(labels=["yes", "no"], **kwargs),
    lambda **kwargs: NaceDocumentClassifier(questions={"q": {"type": "noul", "instructions": "ok?"}}, **kwargs),
    NaceFaithfulnessEvaluator,
    NaceContextRelevanceEvaluator,
]
ANSWER = {
    "model": "drex-max-v1.1",
    "usage": {"input_tokens": 1, "output_tokens": 0},
    "answers": {"q": {"type": "noul", "noul": 0.9}},
}


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    "base_url", ["https://provider.example", "http://console.nace.ai", "https://console.nace.ai:444"]
)
def test_components_refuse_ambient_nace_credentials_at_other_origins(factory, base_url, monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "fake-nace-key")
    monkeypatch.setenv("NACE_BASE_URL", base_url)
    with pytest.raises(ValueError, match=r"explicit.*api_key"):
        factory()
    with pytest.raises(ValueError, match="NACE_API_KEY"):
        factory(api_key=Secret.from_env_var("NACE_API_KEY"), api_base_url=base_url)


@pytest.mark.parametrize("factory", FACTORIES)
def test_saved_pipeline_pins_origin_and_rejects_retargeted_ambient_key(factory, monkeypatch):
    monkeypatch.delenv("NACE_BASE_URL", raising=False)
    pipeline = Pipeline()
    pipeline.add_component("model", factory())
    saved = pipeline.dumps()
    monkeypatch.setenv("NACE_BASE_URL", "https://provider.example")
    restored = Pipeline.loads(saved).get_component("model")
    assert restored.api_base_url == "https://console.nace.ai"
    assert pipeline.get_component("model").api_base_url == "https://console.nace.ai"
    with pytest.raises(DeserializationError, match="NACE_API_KEY"):
        Pipeline.loads(saved.replace("https://console.nace.ai", "https://provider.example"))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("trusted", [False, True])
async def test_actual_http_requests_keep_bypass_only_on_nace(asynchronous, trusted, monkeypatch):
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "fake-bypass-secret")
    monkeypatch.setenv("PROVIDER_API_KEY", "fake-provider-key")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    options = client_options(
        Secret.from_env_var("PROVIDER_API_KEY"),
        "https://console.nace.ai" if trusted else "https://provider.example",
        None,
        0,
    ) | {"transport": httpx.MockTransport(handler)}
    kwargs = {"state": "hi", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    if asynchronous:
        client = make_async_client(options)
        await client.system_one(**kwargs)
        await client.aclose()
    else:
        client = make_client(options)
        client.system_one(**kwargs)
        client.close()
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer fake-provider-key"
    assert seen[0].headers.get("x-vercel-protection-bypass") == ("fake-bypass-secret" if trusted else None)


def test_empty_key_cannot_fall_back_to_sdk_environment(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "fake-nace-key")
    monkeypatch.setenv("PROVIDER_API_KEY", " ")
    with pytest.raises(ValueError, match="empty"):
        client_options(Secret.from_env_var("PROVIDER_API_KEY"), "https://provider.example", None, 0)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_redirect_cannot_forward_credentials_to_another_origin(asynchronous, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://other.example/v1/system-one"})

    original_sync, original_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_sync(follow_redirects=True, **kwargs))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_async(follow_redirects=True, **kwargs))
    options = client_options(Secret.from_token("fake-provider-key"), "https://provider.example", None, 0)
    options["transport"] = httpx.MockTransport(handler)
    kwargs = {"state": "hi", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    client = make_async_client(options) if asynchronous else make_client(options)
    try:
        with pytest.raises(ValueError, match="different API origin"):
            if asynchronous:
                await client.system_one(**kwargs)
            else:
                client.system_one(**kwargs)
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()
    assert len(seen) == 1
    assert seen[0].url.host == "provider.example"
