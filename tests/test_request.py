# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json

import nace_sdk
import pytest
from pydantic import BaseModel

from haystack_integrations.common.nace.images import MAX_REQUEST_BYTES
from haystack_integrations.common.nace.request import DrexResponse, build_call, to_wire, validate_questions

from .helpers import BLUE_PNG, RED_PNG, first_pixel

NOUL = {"type": "noul", "instructions": "Is this about billing?"}


class TestToWire:
    def test_noul_without_instructions_sends_empty_string(self):
        assert to_wire({"type": "noul", "criteria": {"true": "a refund"}}) == {
            "type": "noul",
            "instructions": "",
            "criteria": {"true": "a refund"},
        }
        assert to_wire(nace_sdk.Noul(criteria={"true": "a refund"}))["instructions"] == ""

    def test_noul_unset_criteria_side_dropped(self):
        assert to_wire(nace_sdk.Noul(instructions="x", criteria={"true": "a refund"})) == {
            "type": "noul",
            "instructions": "x",
            "criteria": {"true": "a refund"},
        }

    def test_choice_null_description_kept(self):
        wire = to_wire({"type": "choice", "instructions": "x", "criteria": {"a": None, "b": "B"}})
        assert wire["criteria"] == {"a": None, "b": "B"}
        sdk = to_wire(nace_sdk.Choice(instructions="x", criteria={"a": None, "b": "B"}))
        assert sdk["criteria"] == {"a": None, "b": "B"}

    def test_score_criteria_order_kept(self):
        assert to_wire({"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]})["criteria"] == [
            "low",
            "mid",
            "high",
        ]

    def test_unknown_question_type_passes(self):
        validate_questions({"logo": {"type": "box", "instructions": "Find the logo"}})
        assert to_wire({"type": "box", "instructions": None, "max_boxes": 3}) == {
            "type": "box",
            "instructions": "",
            "max_boxes": 3,
        }


class TestValidateQuestions:
    @pytest.mark.parametrize(
        ("questions", "match"),
        [
            ({}, "at least one question"),
            ({"q": {"instructions": "no type"}}, "Question 'q' needs a `type`"),
            ({"q": {"type": "choice", "instructions": "x"}}, "Question 'q' of type 'choice' needs `criteria`"),
            ({"q": {"type": "score", "instructions": "x", "criteria": []}}, "of type 'score' needs `criteria`"),
        ],
    )
    def test_structure_errors(self, questions, match):
        with pytest.raises(ValueError, match=match):
            validate_questions(questions)

    def test_sdk_questions_and_noul_without_criteria_pass(self):
        validate_questions({"a": nace_sdk.Choice(instructions="x", criteria={"a": None}), "b": NOUL})


class TestBuildCall:
    def test_images_sent_as_data_urls(self):
        call = build_call("t", {"q": NOUL}, model="drex-max", images=[RED_PNG], openrouter=False)
        assert call["model"] == "drex-max"
        assert call["images"][0].startswith("data:image/png;base64,")
        assert call["questions"] == {"q": NOUL}

    def test_text_only_call_has_no_images(self):
        assert build_call("t", {"q": NOUL}, model="drex-v2.0", images=None, openrouter=False)["images"] is None

    def test_two_images_keep_order(self):
        red_first = build_call("t", {"q": NOUL}, model="drex-max", images=[RED_PNG, BLUE_PNG], openrouter=False)
        blue_first = build_call("t", {"q": NOUL}, model="drex-max", images=[BLUE_PNG, RED_PNG], openrouter=False)
        assert first_pixel(red_first["images"][0]) == (255, 0, 0)
        assert first_pixel(blue_first["images"][0]) == (0, 0, 255)

    @pytest.mark.parametrize("model", ["drex-v2.0", "drex-v1.5", "nace-ai/drex-v2.0"])
    def test_images_with_text_only_model_refused(self, model):
        with pytest.raises(ValueError, match="reads text only"):
            build_call("t", {"q": NOUL}, model=model, images=[RED_PNG], openrouter=model.startswith("nace-ai/"))

    def test_unknown_model_with_images_is_sent(self):
        call = build_call("t", {"q": NOUL}, model="drex-max-2", images=[RED_PNG], openrouter=False)
        assert call["model"] == "drex-max-2"
        assert len(call["images"]) == 1

    def test_openrouter_needs_openrouter_model_id(self):
        with pytest.raises(ValueError, match="nace-ai/drex-max"):
            build_call("t", {"q": NOUL}, model="drex-max", images=None, openrouter=True)

    def test_openrouter_moves_images_into_state(self):
        call = build_call("t", {"q": NOUL}, model="nace-ai/drex-max", images=[RED_PNG], openrouter=True)
        assert call["images"] is None
        assert call["state"][0] == {"type": "text", "text": "t"}
        assert call["state"][1]["type"] == "image_url"
        assert call["state"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_openrouter_json_state_becomes_text_part(self):
        call = build_call({"a": 1}, {"q": NOUL}, model="nace-ai/drex-max", images=[RED_PNG], openrouter=True)
        assert call["state"][0] == {"type": "text", "text": '{"a": 1}'}

    def test_big_image_is_shrunk_inside_build_call(self, phone_scan):
        call = build_call("t", {"q": NOUL}, model="drex-max", images=[phone_scan], openrouter=False)
        body = {key: value for key, value in call.items() if key != "response_model"}
        assert len(json.dumps(body)) <= MAX_REQUEST_BYTES
        assert call["images"][0].startswith("data:image/jpeg;base64,")

    def test_build_call_passes_response_model(self):
        call = build_call("t", {"q": NOUL}, model="drex-max", images=None, openrouter=False)
        assert call["response_model"] is DrexResponse


def test_response_keeps_new_types_and_fields():
    response = DrexResponse.model_validate(
        {
            "model": "drex-max",
            "usage": {"input_tokens": 5, "output_tokens": 0, "new": 1},
            "answers": {
                "n": {"type": "noul", "noul": 0.9, "new_field": "x"},
                "b": {"type": "box", "boxes": [[0, 0, 1, 1]]},
            },
            "evaluation_time_ms": 12,
            "request_id": "req_1",
        }
    )
    assert response.answers["b"]["boxes"] == [[0, 0, 1, 1]]
    assert response.answers["n"]["new_field"] == "x"
    assert response.usage["new"] == 1
    assert response.model_extra["request_id"] == "req_1"


REFERENCE = {"type": "workspace_file", "workspace_id": "w", "file_id": "f"}


class _OtherSdkChoice(BaseModel):
    """Shaped like typesafe_sdk.Choice, so Jev code that builds question objects keeps working."""

    type: str = "choice"
    instructions: str | None = None
    criteria: dict[str, str | None]


class TestLimitsAndOtherSdks:
    def test_huge_text_leaves_no_room_for_images(self):
        with pytest.raises(ValueError, match="too big to send with this much text"):
            build_call(
                "x" * (MAX_REQUEST_BYTES + 10), {"q": NOUL}, model="drex-max", images=[RED_PNG], openrouter=False
            )

    def test_huge_text_with_file_reference_only_is_sent(self):
        call = build_call(
            "x" * (MAX_REQUEST_BYTES + 10), {"q": NOUL}, model="drex-max", images=[REFERENCE], openrouter=False
        )
        assert call["images"] == [REFERENCE]

    def test_other_sdk_question_objects_accepted(self):
        question = _OtherSdkChoice(instructions="x", criteria={"a": None, "b": "B"})
        validate_questions({"q": question})
        assert to_wire(question) == {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": "B"}}

    def test_non_question_objects_refused(self):
        with pytest.raises(ValueError, match="Question 'q' must be a dict or a question object"):
            validate_questions({"q": 42})
