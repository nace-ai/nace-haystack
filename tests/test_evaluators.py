# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import nace_sdk
import pytest
from haystack import Pipeline
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret

from haystack_integrations.common.nace.request import DrexResponse
from haystack_integrations.components.evaluators.nace import (
    CONTEXT_RELEVANCE_INSTRUCTIONS,
    FAITHFULNESS_INSTRUCTIONS,
    NaceContextRelevanceEvaluator,
    NaceFaithfulnessEvaluator,
)

from .helpers import RED_PNG

SCORES = {"Who won?": 0.92, "When?": 0.10}


def _verdict(**kwargs) -> DrexResponse:
    """A fake system_one: the P(yes) for each question is looked up in SCORES."""
    question = kwargs["state"]["question"]
    if question == "fail":
        raise nace_sdk.NaceUnprocessableEntityError(
            422, {"error": {"message": "the request is invalid", "issues": []}}, httpx.Headers()
        )
    return DrexResponse.model_validate(
        {
            "model": kwargs["model"],
            "answers": {"verdict": {"type": "noul", "noul": SCORES[question]}},
            "usage": {"input_tokens": 40, "output_tokens": 0},
            "request_id": f"req_{question}",
        }
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


def _faithfulness(**kwargs) -> NaceFaithfulnessEvaluator:
    evaluator = NaceFaithfulnessEvaluator(**kwargs)
    evaluator._client = MagicMock()
    evaluator._client.system_one.side_effect = _verdict
    return evaluator


@pytest.mark.usefixtures("api_key")
class TestFaithfulness:
    def test_outputs_match_haystack_shape(self):
        result = _faithfulness().run(
            questions=["Who won?", "When?"],
            contexts=[["Ada won the race."], ["The race was in May."]],
            predicted_answers=["Ada won.", "In June."],
        )
        assert set(result) == {"score", "individual_scores", "results", "meta"}
        assert result["individual_scores"] == [0.92, 0.10]
        assert result["score"] == pytest.approx(0.51)
        assert result["results"][0] == {"score": 0.92, "noul": 0.92, "status": "evaluated"}
        assert result["meta"][1] == {
            "model": "drex-max-v1.1",
            "usage": {"input_tokens": 40, "output_tokens": 0},
            "request_id": "req_When?",
        }

    def test_state_and_question_sent(self):
        evaluator = _faithfulness()
        evaluator.run(questions=["Who won?"], contexts=[["Ada won the race."]], predicted_answers=["Ada won."])
        call = evaluator._client.system_one.call_args.kwargs
        assert call["state"] == {"question": "Who won?", "contexts": ["Ada won the race."], "answer": "Ada won."}
        assert call["questions"] == {"verdict": {"type": "noul", "instructions": FAITHFULNESS_INSTRUCTIONS}}
        assert call["model"] == "drex-max-v1.1"

    def test_instructions_overridable(self):
        evaluator = _faithfulness(instructions="Is the answer grounded?")
        evaluator.run(questions=["Who won?"], contexts=[["x"]], predicted_answers=["y"])
        assert evaluator._client.system_one.call_args.kwargs["questions"]["verdict"]["instructions"] == (
            "Is the answer grounded?"
        )

    def test_failure_nan_when_not_raising(self, caplog):
        result = _faithfulness(raise_on_failure=False).run(
            questions=["Who won?", "fail"], contexts=[["x"], ["y"]], predicted_answers=["a", "b"]
        )
        assert result["individual_scores"][0] == 0.92
        assert math.isnan(result["individual_scores"][1])
        assert result["results"][1]["status"] == "error"
        assert result["results"][1]["error"].startswith("422 from Drex")
        assert result["score"] == 0.92
        assert "1 item(s) failed" in caplog.text

    def test_failure_raises_by_default(self):
        with pytest.raises(nace_sdk.NaceUnprocessableEntityError, match="422 from Drex"):
            _faithfulness().run(questions=["fail"], contexts=[["y"]], predicted_answers=["b"])

    def test_mismatched_lengths_refused(self):
        evaluator = _faithfulness()
        with pytest.raises(ValueError, match="same length"):
            evaluator.run(questions=["Who won?", "When?"], contexts=[["x"]], predicted_answers=["a", "b"])
        with pytest.raises(ValueError, match="same length"):
            evaluator.run(questions=["Who won?"], contexts=[["x"]], predicted_answers=["a"], images=[[], []])
        evaluator._client.system_one.assert_not_called()

    def test_empty_inputs(self):
        evaluator = _faithfulness()
        result = evaluator.run(questions=[], contexts=[], predicted_answers=[])
        assert math.isnan(result["score"])
        assert result["individual_scores"] == []
        assert result["results"] == []
        assert result["meta"] == []
        evaluator._client.system_one.assert_not_called()

    def test_images_per_item(self):
        evaluator = _faithfulness()
        evaluator.run(
            questions=["Who won?", "When?"],
            contexts=[["See Image 1."], ["x"]],
            predicted_answers=["Ada.", "May."],
            images=[[RED_PNG], []],
        )
        calls = sorted(evaluator._client.system_one.call_args_list, key=lambda call: call.kwargs["state"]["question"])
        when, who = calls
        assert who.kwargs["images"] == [f"data:image/png;base64,{RED_PNG.base64_image}"]
        assert when.kwargs["images"] is None
        assert who.kwargs["model"] == when.kwargs["model"] == "drex-max-v1.1"

    def test_images_with_text_only_model_refused(self):
        evaluator = _faithfulness(model="drex-v2.0")
        with pytest.raises(ValueError, match="reads text only"):
            evaluator.run(questions=["Who won?"], contexts=[["x"]], predicted_answers=["a"], images=[[RED_PNG]])
        evaluator._client.system_one.assert_not_called()

    async def test_run_async(self):
        evaluator = NaceFaithfulnessEvaluator()
        evaluator._async_client = MagicMock()
        evaluator._async_client.system_one = AsyncMock(side_effect=_verdict)
        result = await evaluator.run_async(
            questions=["Who won?", "When?"], contexts=[["x"], ["y"]], predicted_answers=["a", "b"]
        )
        assert result["individual_scores"] == [0.92, 0.10]
        assert evaluator._async_client.system_one.await_count == 2

    def test_same_sockets_as_haystack_builtin(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "unused")
        builtin = FaithfulnessEvaluator()
        ours = NaceFaithfulnessEvaluator()
        assert set(builtin.__haystack_input__._sockets_dict) <= set(ours.__haystack_input__._sockets_dict)
        assert set(builtin.__haystack_output__._sockets_dict) == set(ours.__haystack_output__._sockets_dict)

    def test_runs_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("evaluator", _faithfulness())
        result = pipeline.run(
            {"evaluator": {"questions": ["Who won?"], "contexts": [["x"]], "predicted_answers": ["Ada."]}}
        )
        assert result["evaluator"]["individual_scores"] == [0.92]

    def test_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("OTHER_NACE_KEY", "other")
        evaluator = NaceFaithfulnessEvaluator(
            instructions="Grounded?",
            model="drex-v2.0",
            api_key=Secret.from_env_var("OTHER_NACE_KEY"),
            api_base_url="https://drex.example.com",
            timeout=90.0,
            max_retries=5,
            max_workers=4,
            raise_on_failure=False,
        )
        data = evaluator.to_dict()
        assert data["init_parameters"] == {
            "instructions": "Grounded?",
            "model": "drex-v2.0",
            "api_key": {"env_vars": ["OTHER_NACE_KEY"], "strict": True, "type": "env_var"},
            "api_base_url": "https://drex.example.com",
            "timeout": 90.0,
            "max_retries": 5,
            "max_workers": 4,
            "raise_on_failure": False,
        }
        restored = component_from_dict(NaceFaithfulnessEvaluator, data, name="evaluator")
        assert restored.to_dict() == data
        assert restored.api_key.resolve_value() == "other"


@pytest.mark.usefixtures("api_key")
class TestContextRelevance:
    def test_state_has_no_answer(self):
        evaluator = NaceContextRelevanceEvaluator()
        evaluator._client = MagicMock()
        evaluator._client.system_one.side_effect = _verdict
        result = evaluator.run(questions=["Who won?"], contexts=[["Ada won the race."]])
        call = evaluator._client.system_one.call_args.kwargs
        assert call["state"] == {"question": "Who won?", "contexts": ["Ada won the race."]}
        assert call["questions"]["verdict"]["instructions"] == CONTEXT_RELEVANCE_INSTRUCTIONS
        assert result["individual_scores"] == [0.92]

    def test_same_sockets_as_haystack_builtin(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "unused")
        builtin = ContextRelevanceEvaluator()
        ours = NaceContextRelevanceEvaluator()
        assert set(builtin.__haystack_input__._sockets_dict) <= set(ours.__haystack_input__._sockets_dict)
        assert set(builtin.__haystack_output__._sockets_dict) == set(ours.__haystack_output__._sockets_dict)

    def test_serialization_round_trip(self):
        evaluator = NaceContextRelevanceEvaluator(max_workers=2)
        data = evaluator.to_dict()
        assert data["init_parameters"]["instructions"] == CONTEXT_RELEVANCE_INSTRUCTIONS
        assert component_from_dict(NaceContextRelevanceEvaluator, data, name="evaluator").to_dict() == data


@pytest.mark.usefixtures("api_key")
class TestContextsShape:
    def test_flat_contexts_refused(self):
        evaluator = _faithfulness()
        with pytest.raises(ValueError, match="list of lists of strings"):
            evaluator.run(questions=["Who won?"], contexts=["Ada won the race."], predicted_answers=["Ada."])
        evaluator._client.system_one.assert_not_called()

    def test_flat_contexts_refused_for_relevance(self):
        evaluator = NaceContextRelevanceEvaluator()
        evaluator._client = MagicMock()
        with pytest.raises(ValueError, match="list of lists of strings"):
            evaluator.run(questions=["Who won?"], contexts=["Ada won the race."])
        evaluator._client.system_one.assert_not_called()


@pytest.mark.usefixtures("api_key")
class TestFailFast:
    def test_raise_on_failure_stops_sending_while_an_earlier_request_is_slow(self):
        def verdict(**kwargs) -> DrexResponse:
            if kwargs["state"]["question"] == "Who won?":
                time.sleep(0.5)
            return _verdict(**kwargs)

        evaluator = _faithfulness(raise_on_failure=True, max_workers=2)
        evaluator._client.system_one.side_effect = verdict
        questions = ["Who won?", "fail"] + ["When?"] * 18
        with pytest.raises(nace_sdk.NaceError):
            evaluator.run(questions=questions, contexts=[["c"]] * 20, predicted_answers=["a"] * 20)
        assert evaluator._client.system_one.call_count <= 3

    def test_bad_image_on_a_late_item_fails_before_any_request(self):
        # Every item is encoded before the first request, so a bad input never costs the caller billed requests
        evaluator = _faithfulness()
        with pytest.raises(ValueError, match="Unsupported image"):
            evaluator.run(
                questions=["Who won?"] * 5,
                contexts=[["c"]] * 5,
                predicted_answers=["a"] * 5,
                images=[[], [], [], [], [object()]],
            )
        evaluator._client.system_one.assert_not_called()

    def test_text_only_model_with_images_on_a_late_item_fails_before_any_request(self):
        evaluator = _faithfulness(model="drex-v2.0")
        with pytest.raises(ValueError, match="reads text only"):
            evaluator.run(
                questions=["Who won?"] * 3,
                contexts=[["c"]] * 3,
                predicted_answers=["a"] * 3,
                images=[[], [], [RED_PNG]],
            )
        evaluator._client.system_one.assert_not_called()


@pytest.mark.usefixtures("api_key")
class TestEventLoops:
    @patch("haystack_integrations.components.evaluators.nace._base.make_async_client")
    def test_a_new_event_loop_gets_a_new_client(self, mock_make_client):
        first, second = AsyncMock(), AsyncMock()
        first.system_one.side_effect = _verdict
        second.system_one.side_effect = _verdict
        mock_make_client.side_effect = [first, second]
        evaluator = NaceFaithfulnessEvaluator()
        inputs = {"questions": ["Who won?"], "contexts": [["Ada won."]], "predicted_answers": ["Ada won."]}
        asyncio.run(evaluator.run_async(**inputs))
        asyncio.run(evaluator.run_async(**inputs))
        assert mock_make_client.call_count == 2
        assert second.system_one.await_count == 1


@pytest.mark.usefixtures("api_key")
class TestImagesShape:
    def test_a_flat_images_list_is_refused_clearly(self):
        evaluator = _faithfulness()
        with pytest.raises(ValueError, match=r"one list of images per item"):
            evaluator.run(questions=["Who won?"], contexts=[["c"]], predicted_answers=["a"], images=[RED_PNG])
        evaluator._client.system_one.assert_not_called()
