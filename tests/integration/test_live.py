# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""Live checks against prod Drex (console.nace.ai). Synthetic data only; run with `hatch run test:integration`."""

import base64
import os
import uuid
from pathlib import Path

import nace_sdk
import pytest
from haystack import Document, Pipeline
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.nace.client import DEFAULT_MODEL, client_options, make_client
from haystack_integrations.common.nace.request import build_call
from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier
from haystack_integrations.components.evaluators.nace import NaceFaithfulnessEvaluator
from haystack_integrations.components.routers.nace import NaceTextRouter

from ..helpers import BLUE_PNG, RED_PNG, solid_png

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("NACE_API_KEY"), reason="Set NACE_API_KEY to run the live tests."),
]

FIXTURES = Path(__file__).parents[1] / "fixtures"
EMAIL = "Hi, I was charged twice for my March invoice. Please refund the duplicate charge today. Thanks, Alex"
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {"billing": None, "technical": None, "sales": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this email?",
        "criteria": ["can wait", "this week", "today"],
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}
COLOURS = {"red": "The image is red", "blue": "The image is blue"}


def _classify(questions: dict, model: str = DEFAULT_MODEL) -> dict:
    classifier = NaceDocumentClassifier(questions=questions, model=model, raise_on_failure=True)
    return classifier.run(documents=[Document(content=EMAIL)])["documents"][0].meta["nace"]


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-max-v1.1"])
@pytest.mark.parametrize("question_id", ["department", "urgency", "refund"])
def test_single_question_types(model, question_id):
    answer = _classify({question_id: QUESTIONS[question_id]}, model=model)[question_id]
    if question_id == "department":
        assert answer["choice"] == "billing"
        assert set(answer["probabilities"]) == {"billing", "technical", "sales"}
    elif question_id == "urgency":
        assert set(answer["probabilities"]) == {"0", "1", "2"}
        assert answer["score"] > 1.0
    else:
        assert answer["noul"] > 0.5


def test_mixed_batch_one_request():
    client = make_client(client_options(Secret.from_env_var("NACE_API_KEY"), None, None, None))
    response = client.system_one(**build_call(EMAIL, QUESTIONS, model=DEFAULT_MODEL, images=None, openrouter=False))
    assert set(response.answers) == {"department", "urgency", "refund"}
    assert response.usage["input_tokens"] > 0
    client.close()


def test_default_model_live():
    router = NaceTextRouter(labels={"billing": None, "technical": None})
    router.warm_up()
    real = router._client.system_one
    seen = []

    def capture(**kwargs):
        seen.append(real(**kwargs))
        return seen[-1]

    router._client.system_one = capture
    assert router.run(text=EMAIL) == {"billing": EMAIL}
    assert seen[0].model == DEFAULT_MODEL


def test_noul_criteria_no_instructions_and_null_choice_description():
    answers = _classify(
        {
            "refund": {"type": "noul", "criteria": {"true": "The customer wants money back"}},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": None, "technical": "Bugs"},
            },
        }
    )
    assert answers["refund"]["noul"] > 0.5
    assert answers["team"]["choice"] == "billing"


def test_too_many_questions_refused_clearly():
    # No limit is hardcoded in this package. Drex Max took 16 questions until 2026-10-29 and 512 since, and
    # nace-sdk refuses more than 512 before sending, so this asks for more than any model takes.
    many = {f"q{i}": {"type": "noul", "instructions": f"Is statement {i} about money?"} for i in range(513)}
    with pytest.raises(nace_sdk.NaceError, match=r"(?i)at most 512"):
        _classify(many)


def test_one_image_on_max():
    router = NaceTextRouter(labels=COLOURS, instructions="What colour is Image 1?")
    assert router.run(text="Look at Image 1.", images=[RED_PNG]) == {"red": "Look at Image 1."}


@pytest.mark.parametrize(("images", "expected"), [([RED_PNG, BLUE_PNG], "red"), ([BLUE_PNG, RED_PNG], "blue")])
def test_two_images_order(images, expected):
    router = NaceTextRouter(labels=COLOURS, instructions="What colour is Image 1?")
    assert expected in router.run(text="Two images.", images=images)


def test_images_on_v2_refused_client_side():
    router = NaceTextRouter(labels=COLOURS, model="drex-v2.0")
    with pytest.raises(ValueError, match="reads text only"):
        router.run(text="Look.", images=[RED_PNG])
    assert router._client is None  # refused before a client or request existed


def test_unsupported_image_type_refused_by_drex():
    gif = ByteStream(
        data=base64.b64decode("R0lGODlhAQABAIAAAP8AAAAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="), mime_type="image/gif"
    )
    router = NaceTextRouter(labels=COLOURS)
    with pytest.raises(nace_sdk.NaceUnprocessableEntityError, match=r"images\.0"):
        router.run(text="Look.", images=[gif])


def test_pdf_as_image_gets_converter_hint():
    pdf = ByteStream(data=b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF", mime_type="application/pdf")
    router = NaceTextRouter(labels=COLOURS)
    with pytest.raises(nace_sdk.NaceUnprocessableEntityError, match="PDFToImageContent"):
        router.run(text="Look.", images=[pdf])


def test_phone_scan_shrunk_and_read(phone_scan, caplog):
    router = NaceTextRouter(
        labels={"total_4212": "$4,212.00", "total_2106": "$2,106.00", "total_1042": "$1,042.00"},
        instructions="What total amount is due on the invoice in Image 1?",
    )
    assert "total_4212" in router.run(text="A customer sent this scanned invoice.", images=[phone_scan])
    assert "shrunk" in caplog.text


def test_evaluators_live():
    evaluator = NaceFaithfulnessEvaluator()
    result = evaluator.run(
        questions=["Who won the race?", "Who won the race?"],
        contexts=[["Ada Example won the 2026 city race."], ["Ada Example won the 2026 city race."]],
        predicted_answers=["Ada Example won the race.", "Ben Example won the race by a minute."],
    )
    faithful, unfaithful = result["individual_scores"]
    assert faithful > 0.5 > unfaithful


@pytest.fixture
def uploaded(tmp_path):
    with nace_sdk.NaceClient(api_key=os.environ["NACE_API_KEY"]) as client:
        tag = uuid.uuid4().hex[:8]
        files = {}
        for colour, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            path = tmp_path / f"{tag}-{colour}.png"
            path.write_bytes(solid_png(rgb, size=64))
            files[colour] = client.documents.upload(path, path=f"nace-haystack-tests/{tag}-{colour}.png")
        try:
            yield files
        finally:
            for file in files.values():
                client.documents.delete_file(file.workspace_id, file.file_id)


def test_uploaded_file_reference(uploaded):
    red, blue = uploaded["red"], uploaded["blue"]
    router = NaceTextRouter(labels=COLOURS, instructions="What colour is Image 1?")
    blue_ref = {"type": "workspace_file", "workspace_id": str(blue.workspace_id), "file_id": str(blue.file_id)}
    assert "red" in router.run(text="Uploaded images.", images=[red, blue_ref])
    assert "blue" in router.run(text="Uploaded images.", images=[blue_ref, red])


def test_switched_yaml_live():
    pipeline = Pipeline.loads((FIXTURES / "nace_pipeline.yaml").read_text())
    result = pipeline.run({"router": {"text": EMAIL}, "classifier": {"documents": [Document(content=EMAIL)]}})
    assert result["router"] == {"billing": EMAIL}
    assert result["classifier"]["documents"][0].meta["typesafe"]["department"]["choice"] == "billing"


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="Set OPENROUTER_API_KEY for the OpenRouter checks."
)
class TestOpenRouter:
    def test_text(self):
        router = NaceTextRouter(
            labels={"billing": None, "technical": None},
            model="nace-ai/drex-v2.0",
            api_key=Secret.from_env_var("OPENROUTER_API_KEY"),
            api_base_url="https://openrouter.ai/api/v1",
        )
        assert router.run(text=EMAIL) == {"billing": EMAIL}

    def test_image(self):
        router = NaceTextRouter(
            labels=COLOURS,
            instructions="What colour is Image 1?",
            model="nace-ai/drex-max",
            api_key=Secret.from_env_var("OPENROUTER_API_KEY"),
            api_base_url="https://openrouter.ai/api/v1",
        )
        assert "red" in router.run(text="Look at Image 1.", images=[RED_PNG])
