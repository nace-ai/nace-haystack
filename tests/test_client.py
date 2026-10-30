# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import nace_sdk
import pytest
from haystack.utils import Secret

from haystack_integrations.common.nace.client import (
    CLIENT_NAME,
    MAX_RETRY_WAIT,
    CappedRetryPolicy,
    client_options,
    is_openrouter,
    is_text_only,
    make_client,
    provider_name,
    resolve_base_url,
)

NOUL_ANSWER = {
    "model": "drex-max",
    "usage": {"input_tokens": 12, "output_tokens": 0},
    "answers": {"q": {"type": "noul", "noul": 0.9}},
}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    return Secret.from_env_var("NACE_API_KEY")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("drex-v2.0", True),
        ("nace-ai/drex-v2.0", True),
        ("drex-v1.5", True),
        ("drex-max", False),
        ("drex-latest", False),
        ("drex-v3.0", False),
    ],
)
def test_is_text_only(model, expected):
    assert is_text_only(model) is expected


class TestBaseUrl:
    def test_blank_env_base_url_means_default(self, monkeypatch):
        monkeypatch.setenv("NACE_BASE_URL", "  ")
        assert resolve_base_url(None) == "https://console.nace.ai"

    def test_env_base_url_used_when_no_argument(self, monkeypatch):
        monkeypatch.setenv("NACE_BASE_URL", "https://drex.example.com")
        assert resolve_base_url(None) == "https://drex.example.com"
        assert resolve_base_url("https://other.example.com") == "https://other.example.com"

    def test_openrouter_base_url_trimmed(self, monkeypatch):
        monkeypatch.delenv("NACE_BASE_URL", raising=False)
        assert resolve_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api"
        assert is_openrouter("https://openrouter.ai/api")
        assert provider_name("https://openrouter.ai/api") == "OpenRouter"
        assert not is_openrouter("https://notopenrouter.ai/api")
        assert is_openrouter("https://OPENROUTER.AI./api")
        assert not is_openrouter(None)
        assert provider_name(None) == "Drex"


class TestClientOptions:
    def test_defaults(self, api_key, monkeypatch):
        monkeypatch.delenv("NACE_BASE_URL", raising=False)
        options = client_options(api_key, None, None, None)
        assert options["api_key"] == "nace_sk_test"
        assert options["base_url"] == "https://console.nace.ai"
        assert options["timeout"] == 60.0
        assert options["client_info"] == CLIENT_NAME
        assert options["retry"].timeout is None
        assert options["retry"].max_retries == 2
        assert "HTTP-Referer" not in options["headers"]

    def test_max_retries_passed(self, api_key):
        assert client_options(api_key, None, None, 5)["retry"].max_retries == 5

    def test_short_timeout_warns(self, api_key, caplog):
        options = client_options(api_key, None, 10.0, None)
        assert options["timeout"] == 10.0
        warnings = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "60" in warnings[0].getMessage()
        assert "55" in warnings[0].getMessage()

    def test_openrouter_headers(self):
        options = client_options(Secret.from_token("provider-key"), "https://openrouter.ai/api/v1", None, None)
        assert options["base_url"] == "https://openrouter.ai/api"
        assert options["headers"]["HTTP-Referer"] == "https://nace.ai"
        assert options["headers"]["X-OpenRouter-Title"] == "Drex for Haystack"


class TestClient:
    @staticmethod
    def _flaky(status: int) -> tuple[httpx.MockTransport, list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(
                    status,
                    headers={"retry-after-ms": "10"},
                    json={"error": {"type": "rate_limit_error", "message": "slow down"}},
                )
            return httpx.Response(200, json=NOUL_ANSWER)

        return httpx.MockTransport(handler), seen

    @pytest.mark.parametrize("status", [429, 529])
    def test_retried_with_attribution(self, api_key, status):
        transport, seen = self._flaky(status)
        client = make_client(client_options(api_key, None, None, None) | {"transport": transport})
        response = client.system_one(
            state="hi", questions={"q": {"type": "noul", "instructions": "ok?"}}, model="drex-max"
        )
        assert response.answers["q"].noul == 0.9
        assert len(seen) == 2
        for request in seen:
            assert request.headers["User-Agent"] == CLIENT_NAME
            assert request.headers["X-Nace-Client"] == CLIENT_NAME
            assert json.loads(request.content)["model"] == "drex-max"


@pytest.mark.parametrize("model", ["nace-ai/drex-v2.0:floor", "nace-ai/drex-v1.5:free", "drex-v2.0:nitro"])
def test_openrouter_variants_of_text_only_models_are_text_only(model):
    assert is_text_only(model)


def test_drex_max_variant_reads_images():
    assert not is_text_only("nace-ai/drex-max:floor")


def test_retry_wait_is_capped():
    error = nace_sdk.NaceRateLimitError(429, None, httpx.Headers({"retry-after-ms": "600000"}), message="slow down")
    assert CappedRetryPolicy()._wait(1, error) == MAX_RETRY_WAIT


def test_short_retry_wait_is_kept():
    error = nace_sdk.NaceRateLimitError(429, None, httpx.Headers({"retry-after-ms": "1500"}), message="slow down")
    assert CappedRetryPolicy()._wait(1, error) == 1.5


def test_client_options_use_the_capped_policy():
    assert isinstance(client_options(Secret.from_token("k"), None, None, None)["retry"], CappedRetryPolicy)
