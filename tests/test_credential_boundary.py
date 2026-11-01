# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import nace_sdk
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
        with pytest.raises((ValueError, nace_sdk.NaceError), match=r"different (API )?origin"):
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


OPENROUTER = "https://openrouter.ai/api/v1"


def _openrouter_request(headers=None, api_key="sk-or-fake"):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    options = client_options(Secret.from_token(api_key), OPENROUTER, None, 0, headers)
    client = make_client(options | {"transport": httpx.MockTransport(handler)})
    client.system_one(state="hi", questions={"q": {"type": "noul", "instructions": "ok?"}})
    client.close()
    return seen[0]


def test_openrouter_requests_carry_the_attribution_defaults():
    request = _openrouter_request()
    assert request.headers.get_list("http-referer") == ["https://nace.ai"]
    assert request.headers.get_list("x-openrouter-title") == ["Drex for Haystack"]


def test_a_lowercase_caller_override_replaces_the_default_with_a_single_value():
    request = _openrouter_request({"http-referer": "https://myapp.example", "x-openrouter-title": "My app"})
    assert request.headers.get_list("http-referer") == ["https://myapp.example"]
    assert request.headers.get_list("x-openrouter-title") == ["My app"]


def test_nace_requests_do_not_carry_the_openrouter_attribution():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    options = client_options(Secret.from_token("nace_fake"), "https://console.nace.ai", None, 0)
    client = make_client(options | {"transport": httpx.MockTransport(handler)})
    client.system_one(state="hi", questions={"q": {"type": "noul", "instructions": "ok?"}})
    client.close()
    assert "http-referer" not in seen[0].headers
    assert "x-openrouter-title" not in seen[0].headers


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    "host", ["https://openrouter.ai/api/v1", "https://eu.openrouter.ai/api/v1", "http://openrouter.ai/api/v1"]
)
@pytest.mark.parametrize("ambient", ["NACE_API_KEY", "OPENROUTER_API_KEY"])
def test_a_nace_key_is_never_sent_to_openrouter(factory, host, ambient, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    monkeypatch.setenv(ambient, "nace_fake")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        factory(api_base_url=host).warm_up()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        factory(api_base_url=host, api_key=Secret.from_env_var(ambient)).warm_up()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        factory(api_base_url=host, api_key=Secret.from_token(" nace_fake")).warm_up()


@pytest.mark.parametrize("factory", FACTORIES)
def test_openrouter_key_is_preferred_over_nace_key(factory, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-first")
    monkeypatch.setenv("NACE_API_KEY", "sk-or-second")
    assert factory(api_base_url=OPENROUTER)._client_options()["api_key"] == "sk-or-first"


@pytest.mark.parametrize("factory", FACTORIES)
def test_an_openrouter_key_in_nace_api_key_still_works(factory, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("NACE_API_KEY", "sk-or-legacy")
    assert factory(api_base_url=OPENROUTER)._client_options()["api_key"] == "sk-or-legacy"
    explicit = factory(api_base_url=OPENROUTER, api_key=Secret.from_env_var("NACE_API_KEY"))
    assert explicit._client_options()["api_key"] == "sk-or-legacy"


@pytest.mark.parametrize("factory", FACTORIES)
def test_an_explicit_key_beats_the_environment(factory, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    component = factory(api_base_url=OPENROUTER, api_key=Secret.from_token("sk-or-explicit"))
    assert component._client_options()["api_key"] == "sk-or-explicit"


@pytest.mark.parametrize("factory", FACTORIES)
def test_saved_openrouter_pipeline_keeps_its_headers_and_key_lookup(factory, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    headers = {"HTTP-Referer": "https://myapp.example", "X-OpenRouter-Title": "My app"}
    pipeline = Pipeline()
    pipeline.add_component("model", factory(api_base_url=OPENROUTER, headers=headers))
    restored = Pipeline.loads(pipeline.dumps()).get_component("model")
    options = restored._client_options()
    assert options["api_key"] == "sk-or-env"
    assert options["headers"] == headers
