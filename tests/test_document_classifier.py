# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

# Ported from deepset's typesafe-haystack tests (integrations/typesafe/tests/test_document_classifier.py),
# with Drex names.

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import nace_sdk
import pytest
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret
from pydantic import BaseModel

from haystack_integrations.common.nace.request import DrexResponse
from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier

COMPONENT_TYPE = "haystack_integrations.components.classifiers.nace.document_classifier.NaceDocumentClassifier"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {"billing": None, "technical": None},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "today"]},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}


def _response_for(state: str, questions: dict, **_) -> DrexResponse:  # noqa: ARG001
    return _response(choice="billing" if "charged" in state else "technical")


def _response(choice: str) -> DrexResponse:
    # Decoded from JSON like a real response. Drex answers stay plain dicts, so score level keys stay strings
    return DrexResponse.model_validate_json(
        json.dumps(
            {
                "model": "drex-v2.0",
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.9,
                        "probabilities": {"billing": 0.95, "technical": 0.05},
                    },
                    "urgency": {
                        "type": "score",
                        "score": 0.7,
                        "confidence": 0.6,
                        "legend": {"0": "can wait", "1": "today"},
                        "probabilities": {"0": 0.3, "1": 0.7},
                    },
                    "refund": {"type": "noul", "noul": 0.8},
                },
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


@pytest.mark.usefixtures("api_key")
class TestInit:
    def test_init_default(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        assert component.questions == QUESTIONS
        assert component.model == "drex-max-v1.1"
        assert component.api_key == Secret.from_env_var("NACE_API_KEY")
        assert component.api_base_url == "https://console.nace.ai"
        assert component.metadata_field == "nace"
        assert component._client is None
        assert component._async_client is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"questions": {}}, "at least one question"),
            ({"questions": {"q": {"type": "choice", "instructions": "?"}}}, "needs `criteria`"),
            ({"questions": {"q": {"type": "score", "instructions": "?", "criteria": []}}}, "needs `criteria`"),
            ({"questions": QUESTIONS, "max_workers": 0}, "at least 1"),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            NaceDocumentClassifier(**kwargs)

    def test_init_accepts_unknown_question_types(self):
        questions = {"q": {"type": "yesno", "instructions": "?"}}
        component = NaceDocumentClassifier(questions=questions)
        assert component.questions == questions


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OTHER_NACE_KEY", "other")
        component = NaceDocumentClassifier(
            questions=QUESTIONS,
            model="drex-v2.0",
            api_key=Secret.from_env_var("OTHER_NACE_KEY"),
            api_base_url="https://drex.example.com",
            classification_field="summary",
            metadata_field="decisions",
            timeout=90.0,
            max_retries=5,
            max_workers=8,
            raise_on_failure=True,
        )
        data = component.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "questions": QUESTIONS,
                "model": "drex-v2.0",
                "api_key": {"env_vars": ["OTHER_NACE_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://drex.example.com",
                "classification_field": "summary",
                "metadata_field": "decisions",
                "timeout": 90.0,
                "max_retries": 5,
                "max_workers": 8,
                "raise_on_failure": True,
            },
        }
        restored = component_from_dict(NaceDocumentClassifier, data, name="classifier")
        assert restored.questions == QUESTIONS
        assert restored.model == "drex-v2.0"
        assert restored.api_key.resolve_value() == "other"
        assert restored.api_base_url == "https://drex.example.com"
        assert restored.classification_field == "summary"
        assert restored.metadata_field == "decisions"
        assert restored.timeout == 90.0
        assert restored.max_retries == 5
        assert restored.max_workers == 8
        assert restored.raise_on_failure is True

    def test_to_dict_with_question_objects(self):
        component = NaceDocumentClassifier(
            questions={
                "department": nace_sdk.Choice(
                    instructions="Which department?", criteria={"billing": None, "technical": None}
                ),
                "urgency": nace_sdk.Score(instructions="How urgent?", criteria=["can wait", "today"]),
                "refund": nace_sdk.Noul(instructions="Is this a refund request?"),
            }
        )
        # The literals are what the SDK's `model_dump()` returns for these questions
        assert component.to_dict()["init_parameters"]["questions"] == {
            "department": {
                "type": "choice",
                "criteria": {"billing": None, "technical": None},
                "instructions": "Which department?",
            },
            "urgency": {"type": "score", "criteria": ["can wait", "today"], "instructions": "How urgent?"},
            "refund": {"type": "noul", "instructions": "Is this a refund request?"},
        }

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("classifier", NaceDocumentClassifier(questions=QUESTIONS))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("classifier").to_dict() == pipeline.get_component("classifier").to_dict()


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack_integrations.components.classifiers.nace.document_classifier.make_client")
    def test_warm_up(self, mock_make_client):
        component = NaceDocumentClassifier(
            questions=QUESTIONS,
            api_key=Secret.from_token("test-key"),
            api_base_url="https://drex.example.com",
            timeout=90.0,
            max_retries=5,
        )
        component.warm_up()
        component.warm_up()
        mock_make_client.assert_called_once()
        options = mock_make_client.call_args.args[0]
        assert options["api_key"] == "test-key"
        assert options["base_url"] == "https://drex.example.com"
        assert options["timeout"] == 90.0
        assert options["retry"].max_retries == 5
        # The SDK's total retry budget is disabled so a long attempt can still be retried
        assert options["retry"].timeout is None
        assert component._client is mock_make_client.return_value

    @patch("haystack_integrations.components.classifiers.nace.document_classifier.make_async_client")
    async def test_warm_up_async(self, mock_make_client):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        await component.warm_up_async()
        await component.warm_up_async()
        mock_make_client.assert_called_once()
        options = mock_make_client.call_args.args[0]
        assert options["retry"].max_retries == 2
        assert options["retry"].timeout is None
        assert component._async_client is mock_make_client.return_value

    def test_close(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        client = MagicMock()
        component._client = client
        component.close()
        component.close()
        client.close.assert_called_once()
        assert component._client is None

    async def test_close_async(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        client = MagicMock(aclose=AsyncMock())
        component._async_client = client
        await component.close_async()
        await component.close_async()
        client.aclose.assert_awaited_once()
        assert component._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    def test_run(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _response_for
        documents = [
            Document(content="I was charged twice.", meta={"source": "email"}),
            Document(content="The app crashes on startup."),
        ]
        result = component.run(documents=documents)
        assert sorted(call.kwargs["state"] for call in component._client.system_one.call_args_list) == [
            "I was charged twice.",
            "The app crashes on startup.",
        ]
        classified = result["documents"]
        assert classified[0].meta == {
            "source": "email",
            "nace": {
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "probabilities": {"billing": 0.95, "technical": 0.05},
                },
                "urgency": {
                    "type": "score",
                    "score": 0.7,
                    "confidence": 0.6,
                    "legend": {"0": "can wait", "1": "today"},
                    "probabilities": {"0": 0.3, "1": 0.7},
                },
                "refund": {"type": "noul", "noul": 0.8},
            },
        }
        assert classified[1].meta["nace"]["department"]["choice"] == "technical"
        assert result["failed_documents"] == []
        # Input documents are not mutated
        assert "nace" not in documents[0].meta

    def test_run_classification_field_and_documents_without_text(self):
        component = NaceDocumentClassifier(questions=QUESTIONS, classification_field="summary")
        component._client = MagicMock()
        component._client.system_one.return_value = _response("billing")
        documents = [
            Document(content="ignored", meta={"summary": "Double charge"}),
            Document(content="no summary"),
        ]
        result = component.run(documents=documents)
        component._client.system_one.assert_called_once_with(
            state="Double charge", questions=QUESTIONS, model="drex-max-v1.1", images=None, response_model=DrexResponse
        )
        assert [document.meta["nace"]["department"]["choice"] for document in result["documents"]] == ["billing"]
        assert [document.meta for document in result["failed_documents"]] == [
            {"classification_error": "Document has no text to classify."}
        ]

    def test_run_records_failed_requests(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = [nace_sdk.NaceAPIConnectionError("connection refused")]
        result = component.run(documents=[Document(content="I was charged twice.")])
        assert result["documents"] == []
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    def test_run_clears_previous_error(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _response_for
        document = Document(content="I was charged twice.", meta={"classification_error": "connection refused"})
        result = component.run(documents=[document])
        assert "classification_error" not in result["documents"][0].meta

    def test_run_raise_on_failure(self):
        component = NaceDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._client = MagicMock()
        component._client.system_one.side_effect = nace_sdk.NaceAPIConnectionError("connection refused")
        with pytest.raises(nace_sdk.NaceAPIConnectionError, match="connection refused"):
            component.run(documents=[Document(content="I was charged twice.")])

    def test_raise_on_failure_stops_sending_while_an_earlier_request_is_slow(self):
        def answer(state: str, questions: dict, **kwargs) -> DrexResponse:
            if state == "Ticket 0":
                time.sleep(0.5)
            if state == "Ticket 1":
                msg = "connection refused"
                raise nace_sdk.NaceAPIConnectionError(msg)
            return _response_for(state, questions, **kwargs)

        component = NaceDocumentClassifier(questions=QUESTIONS, raise_on_failure=True, max_workers=2)
        component._client = MagicMock()
        component._client.system_one.side_effect = answer
        with pytest.raises(nace_sdk.NaceAPIConnectionError):
            component.run(documents=[Document(content=f"Ticket {number}") for number in range(20)])
        assert component._client.system_one.call_count <= 3

    async def test_run_async(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(side_effect=_response_for)
        documents = [Document(content="I was charged twice."), Document(content=""), Document(content="It crashes.")]
        result = await component.run_async(documents=documents)
        assert component._async_client.system_one.await_count == 2
        choices = [document.meta["nace"]["department"]["choice"] for document in result["documents"]]
        assert choices == ["billing", "technical"]
        assert result["failed_documents"][0].meta == {"classification_error": "Document has no text to classify."}

    async def test_run_async_records_failed_requests(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(
            side_effect=[_response("billing"), nace_sdk.NaceAPIConnectionError("connection refused")]
        )
        documents = [Document(content="I was charged twice."), Document(content="It crashes.")]
        result = await component.run_async(documents=documents)
        assert len(result["documents"]) == 1
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    async def test_run_async_raise_on_failure_cancels_remaining_requests(self):
        cancelled = asyncio.Event()

        async def system_one(state, questions, **_):  # noqa: ARG001
            if state == "fails":
                msg = "connection refused"
                raise nace_sdk.NaceAPIConnectionError(msg)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        component = NaceDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._async_client = MagicMock(system_one=system_one)
        with pytest.raises(nace_sdk.NaceAPIConnectionError, match="connection refused"):
            await component.run_async(documents=[Document(content="slow"), Document(content="fails")])
        assert cancelled.is_set()

    async def test_run_async_limits_concurrent_requests(self):
        in_flight = 0
        max_in_flight = 0

        async def system_one(state, questions, **_):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _response_for(state=state, questions=questions)

        component = NaceDocumentClassifier(questions=QUESTIONS, max_workers=2)
        component._async_client = MagicMock(system_one=system_one)
        documents = [Document(content=f"Ticket {i}") for i in range(6)]
        result = await component.run_async(documents=documents)
        assert max_in_flight == 2
        assert all("nace" in document.meta for document in result["documents"])


def _drex_response(answers: dict) -> DrexResponse:
    return DrexResponse.model_validate({"model": "drex-max", "answers": answers, "usage": {"input_tokens": 9}})


def _unprocessable() -> nace_sdk.NaceUnprocessableEntityError:
    return nace_sdk.NaceUnprocessableEntityError(
        422,
        {"error": {"message": "the request is invalid", "issues": [{"path": "questions.refund", "message": "bad"}]}},
        httpx.Headers(),
    )


@pytest.mark.usefixtures("api_key")
class TestDrex:
    def test_one_request_per_document_with_all_questions(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _response_for
        component.run(documents=[Document(content="Double charge"), Document(content="App crashes")])
        assert component._client.system_one.call_count == 2
        for call in component._client.system_one.call_args_list:
            assert set(call.kwargs["questions"]) == set(QUESTIONS)
            assert call.kwargs["model"] == "drex-max-v1.1"

    def test_noul_without_instructions_on_wire(self):
        component = NaceDocumentClassifier(questions={"refund": {"type": "noul", "criteria": {"true": "money back"}}})
        component._client = MagicMock()
        component._client.system_one.return_value = _drex_response({"refund": {"type": "noul", "noul": 0.8}})
        component.run(documents=[Document(content="Refund please")])
        sent = component._client.system_one.call_args.kwargs["questions"]
        assert sent == {"refund": {"type": "noul", "instructions": "", "criteria": {"true": "money back"}}}

    def test_failure_stores_readable_message(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _unprocessable()
        result = component.run(documents=[Document(content="Double charge")])
        assert result["documents"] == []
        error = result["failed_documents"][0].meta["classification_error"]
        assert error.startswith("422 from Drex: the request is invalid")
        assert "questions.refund: bad" in error

    def test_raise_on_failure_raises_sdk_class(self):
        component = NaceDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._client = MagicMock()
        component._client.system_one.side_effect = _unprocessable()
        with pytest.raises(nace_sdk.NaceUnprocessableEntityError, match="422 from Drex"):
            component.run(documents=[Document(content="Double charge")])

    async def test_raise_on_failure_async_raises_sdk_class(self):
        component = NaceDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(side_effect=_unprocessable())
        with pytest.raises(nace_sdk.NaceUnprocessableEntityError, match="422 from Drex"):
            await component.run_async(documents=[Document(content="Double charge")])

    def test_empty_documents_no_calls(self):
        component = NaceDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        assert component.run(documents=[]) == {"documents": [], "failed_documents": []}
        component._client.system_one.assert_not_called()

    def test_future_answer_type_and_fields_stored(self):
        answers = {
            "logo": {"type": "box", "boxes": [[0, 0, 10, 10]]},
            "refund": {"type": "noul", "noul": 0.9, "new_field": "x"},
        }
        component = NaceDocumentClassifier(
            questions={"logo": {"type": "box", "instructions": "Find the logo"}, "refund": QUESTIONS["refund"]}
        )
        component._client = MagicMock()
        component._client.system_one.return_value = _drex_response(answers)
        result = component.run(documents=[Document(content="Invoice with a logo")])
        assert result["documents"][0].meta["nace"] == answers


class _OtherSdkNoul(BaseModel):
    """Shaped like typesafe_sdk.Noul."""

    type: str = "noul"
    instructions: str | None = None


@pytest.mark.usefixtures("api_key")
def test_other_sdk_question_objects_switch_unchanged():
    component = NaceDocumentClassifier(questions={"refund": _OtherSdkNoul(instructions="Refund?")})
    assert component.to_dict()["init_parameters"]["questions"] == {
        "refund": {"type": "noul", "instructions": "Refund?"}
    }
    component._client = MagicMock()
    component._client.system_one.return_value = _drex_response({"refund": {"type": "noul", "noul": 0.9}})
    result = component.run(documents=[Document(content="Refund please")])
    assert component._client.system_one.call_args.kwargs["questions"] == {
        "refund": {"type": "noul", "instructions": "Refund?"}
    }
    assert result["documents"][0].meta["nace"]["refund"]["noul"] == 0.9


@pytest.mark.usefixtures("api_key")
class TestEventLoops:
    @patch("haystack_integrations.components.classifiers.nace.document_classifier.make_async_client")
    def test_a_new_event_loop_gets_a_new_client(self, mock_make_client):
        first, second = AsyncMock(), AsyncMock()
        first.system_one.side_effect = _response_for
        second.system_one.side_effect = _response_for
        mock_make_client.side_effect = [first, second]
        classifier = NaceDocumentClassifier(questions=QUESTIONS)
        asyncio.run(classifier.run_async(documents=[Document(content="I was charged twice.")]))
        asyncio.run(classifier.run_async(documents=[Document(content="I was charged twice.")]))
        assert mock_make_client.call_count == 2
        assert second.system_one.await_count == 1
