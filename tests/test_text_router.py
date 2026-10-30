# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

# Ported from deepset's typesafe-haystack tests (integrations/typesafe/tests/test_text_router.py), with Drex names,
# plus the Drex additions: images, the probabilities output and readable errors.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import nace_sdk
import pytest
from haystack import Pipeline
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret

from haystack_integrations.common.nace.request import DrexResponse
from haystack_integrations.components.routers.nace import NaceTextRouter

from .helpers import RED_PNG

COMPONENT_TYPE = "haystack_integrations.components.routers.nace.text_router.NaceTextRouter"

LABELS = {"billing": "Payments, invoices and refunds", "technical": "Bugs and errors in the product"}

PROBABILITIES = {"billing": 0.5, "technical": 0.5}


def _response(choice: str, confidence: float = 0.9) -> DrexResponse:
    return DrexResponse.model_validate(
        {
            "model": "drex-max",
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": PROBABILITIES,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    )


def _expected_call(text: str, instructions: str, criteria: dict, *, model: str = "drex-max-v1.1", images=None) -> dict:
    return {
        "state": text,
        "questions": {"route": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        "model": model,
        "images": images,
        "response_model": DrexResponse,
    }


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


@pytest.mark.usefixtures("api_key")
class TestInit:
    def test_init_default(self):
        router = NaceTextRouter(labels=["billing", "technical"])
        assert router.labels == ["billing", "technical"]
        assert router.instructions == "Which category does this text belong to?"
        assert router.model == "drex-max-v1.1"
        assert router.api_key == Secret.from_env_var("NACE_API_KEY")
        assert router.min_confidence is None
        assert router.include_probabilities is False
        assert router._question == nace_sdk.Choice(
            instructions="Which category does this text belong to?", criteria={"billing": None, "technical": None}
        )
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical"}

    def test_init_min_confidence_adds_low_confidence_output(self):
        router = NaceTextRouter(labels=LABELS, min_confidence=0.6)
        assert router._question.criteria == LABELS
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical", "low_confidence"}

    def test_init_include_probabilities_adds_output(self):
        router = NaceTextRouter(labels=LABELS, min_confidence=0.6, include_probabilities=True)
        assert set(router.__haystack_output__._sockets_dict) == {
            "billing",
            "technical",
            "low_confidence",
            "probabilities",
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"labels": []}, "at least one label"),
            ({"labels": ["billing", "low_confidence"]}, "reserved"),
            ({"labels": ["billing"], "min_confidence": 1.5}, "between 0 and 1"),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            NaceTextRouter(**kwargs)

    def test_label_probabilities_reserved_only_when_enabled(self):
        router = NaceTextRouter(labels=["probabilities", "other"])
        assert set(router.__haystack_output__._sockets_dict) == {"probabilities", "other"}
        with pytest.raises(ValueError, match="reserved"):
            NaceTextRouter(labels=["probabilities", "other"], include_probabilities=True)


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OTHER_NACE_KEY", "other")
        router = NaceTextRouter(
            labels=LABELS,
            instructions="Which team should handle this?",
            model="drex-v2.0",
            api_key=Secret.from_env_var("OTHER_NACE_KEY"),
            api_base_url="https://drex.example.com",
            min_confidence=0.6,
            include_probabilities=True,
            timeout=90.0,
            max_retries=5,
        )
        data = router.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "labels": LABELS,
                "instructions": "Which team should handle this?",
                "model": "drex-v2.0",
                "api_key": {"env_vars": ["OTHER_NACE_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://drex.example.com",
                "min_confidence": 0.6,
                "include_probabilities": True,
                "timeout": 90.0,
                "max_retries": 5,
            },
        }
        restored = component_from_dict(NaceTextRouter, data, name="router")
        assert restored.labels == LABELS
        assert restored.instructions == "Which team should handle this?"
        assert restored.model == "drex-v2.0"
        assert restored.api_key.resolve_value() == "other"
        assert restored.api_base_url == "https://drex.example.com"
        assert restored.min_confidence == 0.6
        assert restored.include_probabilities is True
        assert restored.timeout == 90.0
        assert restored.max_retries == 5

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("router", NaceTextRouter(labels=LABELS, min_confidence=0.6))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("router").to_dict() == pipeline.get_component("router").to_dict()


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack_integrations.components.routers.nace.text_router.make_client")
    def test_warm_up(self, mock_make_client):
        router = NaceTextRouter(
            labels=LABELS, api_key=Secret.from_token("test-key"), api_base_url="https://drex.example.com", max_retries=5
        )
        router.warm_up()
        router.warm_up()
        mock_make_client.assert_called_once()
        options = mock_make_client.call_args.args[0]
        assert options["api_key"] == "test-key"
        assert options["base_url"] == "https://drex.example.com"
        assert options["timeout"] == 60.0
        assert options["retry"].max_retries == 5
        assert options["retry"].timeout is None
        assert router._client is mock_make_client.return_value

    @patch("haystack_integrations.components.routers.nace.text_router.make_async_client")
    async def test_warm_up_async(self, mock_make_client):
        router = NaceTextRouter(labels=LABELS)
        await router.warm_up_async()
        await router.warm_up_async()
        mock_make_client.assert_called_once()
        assert router._async_client is mock_make_client.return_value

    def test_close(self):
        router = NaceTextRouter(labels=LABELS)
        client = MagicMock()
        router._client = client
        router.close()
        router.close()
        client.close.assert_called_once()
        assert router._client is None

    async def test_close_async(self):
        router = NaceTextRouter(labels=LABELS)
        client = MagicMock(aclose=AsyncMock())
        router._async_client = client
        await router.close_async()
        await router.close_async()
        client.aclose.assert_awaited_once()
        assert router._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    @pytest.mark.parametrize(
        ("confidence", "expected_output"),
        [(0.9, "billing"), (0.3, "low_confidence")],
    )
    def test_run(self, confidence, expected_output):
        router = NaceTextRouter(labels=LABELS, instructions="Which team should handle this?", min_confidence=0.6)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("billing", confidence=confidence)
        result = router.run(text="I was charged twice.")
        router._client.system_one.assert_called_once_with(
            **_expected_call("I was charged twice.", "Which team should handle this?", LABELS)
        )
        assert result == {expected_output: "I was charged twice."}

    def test_run_without_min_confidence_ignores_confidence(self):
        router = NaceTextRouter(labels=LABELS)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("technical", confidence=0.01)
        assert router.run(text="Hello?") == {"technical": "Hello?"}

    @pytest.mark.parametrize(
        ("confidence", "expected_output"),
        [(0.9, "billing"), (0.3, "low_confidence")],
    )
    async def test_run_async(self, confidence, expected_output):
        router = NaceTextRouter(labels=LABELS, min_confidence=0.6)
        router._async_client = MagicMock()
        router._async_client.system_one = AsyncMock(return_value=_response("billing", confidence=confidence))
        assert await router.run_async(text="I was charged twice.") == {expected_output: "I was charged twice."}

    async def test_run_wrong_input_type(self):
        router = NaceTextRouter(labels=LABELS)
        with pytest.raises(TypeError, match="expects a str"):
            router.run(text=["not", "a", "string"])
        with pytest.raises(TypeError, match="expects a str"):
            await router.run_async(text=["not", "a", "string"])


@pytest.mark.usefixtures("api_key")
class TestDrex:
    def test_default_model_sent(self):
        router = NaceTextRouter(labels=LABELS)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("billing")
        router.run(text="Hi")
        assert router._client.system_one.call_args.kwargs["model"] == "drex-max-v1.1"

    def test_images_route_on_max(self):
        router = NaceTextRouter(labels={"red": None, "blue": None})
        router._client = MagicMock()
        router._client.system_one.return_value = _response("red")
        assert router.run(text="Which colour is Image 1?", images=[RED_PNG]) == {"red": "Which colour is Image 1?"}
        call = router._client.system_one.call_args.kwargs
        assert call["model"] == "drex-max-v1.1"
        assert call["images"] == [f"data:image/png;base64,{RED_PNG.base64_image}"]

    async def test_images_async(self):
        router = NaceTextRouter(labels={"red": None, "blue": None})
        router._async_client = MagicMock()
        router._async_client.system_one = AsyncMock(return_value=_response("red"))
        assert await router.run_async(text="Which colour?", images=[RED_PNG]) == {"red": "Which colour?"}
        assert len(router._async_client.system_one.call_args.kwargs["images"]) == 1

    def test_images_with_text_only_model_refused_before_request(self):
        router = NaceTextRouter(labels=LABELS, model="drex-v2.0")
        router._client = MagicMock()
        with pytest.raises(ValueError, match="reads text only"):
            router.run(text="Hi", images=[RED_PNG])
        router._client.system_one.assert_not_called()

    def test_openrouter_needs_its_model_id(self):
        router = NaceTextRouter(
            labels=LABELS, api_key=Secret.from_token("provider-key"), api_base_url="https://openrouter.ai/api/v1"
        )
        router._client = MagicMock()
        with pytest.raises(ValueError, match="nace-ai/drex-max"):
            router.run(text="Hi")
        router._client.system_one.assert_not_called()

    def test_probabilities_output(self):
        router = NaceTextRouter(labels=LABELS, include_probabilities=True)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("billing", confidence=0.9)
        assert router.run(text="Refund me") == {
            "billing": "Refund me",
            "probabilities": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": PROBABILITIES,
            },
        }

    def test_probabilities_with_low_confidence(self):
        router = NaceTextRouter(labels=LABELS, min_confidence=0.6, include_probabilities=True)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("billing", confidence=0.4)
        result = router.run(text="Hmm")
        assert result["low_confidence"] == "Hmm"
        assert result["probabilities"]["confidence"] == 0.4
        assert "billing" not in result

    def test_api_error_is_readable(self):
        router = NaceTextRouter(labels=LABELS)
        router._client = MagicMock()
        router._client.system_one.side_effect = nace_sdk.NaceUnprocessableEntityError(
            422,
            {
                "error": {
                    "message": "the request is invalid",
                    "issues": [
                        {"path": "questions.route.criteria", "message": "too few labels"},
                        {"path": "state", "message": "too long"},
                    ],
                }
            },
            httpx.Headers(),
        )
        with pytest.raises(nace_sdk.NaceUnprocessableEntityError) as error:
            router.run(text="Hi")
        assert str(error.value).startswith("422 from Drex:")
        assert "questions.route.criteria: too few labels" in str(error.value)


@pytest.mark.usefixtures("api_key")
class TestEventLoops:
    @patch("haystack_integrations.components.routers.nace.text_router.make_async_client")
    def test_a_new_event_loop_gets_a_new_client(self, mock_make_client):
        # An httpx async client is bound to the event loop it first ran on, so reusing it after `asyncio.run`
        # returns fails with "Event loop is closed"
        first, second = AsyncMock(), AsyncMock()
        first.system_one.return_value = _response("billing")
        second.system_one.return_value = _response("billing")
        mock_make_client.side_effect = [first, second]
        router = NaceTextRouter(labels=["billing", "technical"])
        asyncio.run(router.run_async(text="Refund me."))
        asyncio.run(router.run_async(text="Refund me again."))
        assert mock_make_client.call_count == 2
        assert second.system_one.await_count == 1

    @patch("haystack_integrations.components.routers.nace.text_router.make_async_client")
    async def test_the_same_event_loop_reuses_its_client(self, mock_make_client):
        client = AsyncMock()
        client.system_one.return_value = _response("billing")
        mock_make_client.return_value = client
        router = NaceTextRouter(labels=["billing", "technical"])
        await router.run_async(text="Refund me.")
        await router.run_async(text="Refund me again.")
        assert mock_make_client.call_count == 1


@pytest.mark.usefixtures("api_key")
class TestUnexpectedInput:
    def test_a_choice_that_isnt_a_label_raises(self):
        # A compatible host could answer with a label it wasn't given; silently dropping the text would lose it
        router = NaceTextRouter(labels=["billing", "technical"], min_confidence=0.5)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("Billing")
        with pytest.raises(ValueError, match="'Billing', which isn't one of the labels"):
            router.run(text="Refund me.")

    def test_a_label_named_type_with_descriptions_is_refused(self):
        # Haystack reads a saved dict with a "type" key as a serialized object, so the pipeline couldn't be loaded
        with pytest.raises(ValueError, match="`type`"):
            NaceTextRouter(labels={"type": "Questions about the product type, e.g. size.", "price": "Pricing"})

    def test_a_label_named_type_in_a_list_round_trips(self):
        pipeline = Pipeline()
        pipeline.add_component("router", NaceTextRouter(labels=["type", "price"]))
        loaded = Pipeline.loads(pipeline.dumps())
        assert loaded.get_component("router").labels == ["type", "price"]
