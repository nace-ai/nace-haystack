# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""The switching contract: a Jev (typesafe-haystack) pipeline runs on Drex after the documented edits only."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline

from haystack_integrations.common.nace.request import DrexResponse
from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier
from haystack_integrations.components.routers.nace import NaceTextRouter

FIXTURES = Path(__file__).parent / "fixtures"

# The edits the README's "Switching from TypeSafe" section documents for a saved YAML pipeline
DOCUMENTED_YAML_EDITS = [
    (
        "haystack_integrations.components.routers.typesafe.text_router.TypeSafeTextRouter",
        "haystack_integrations.components.routers.nace.text_router.NaceTextRouter",
    ),
    (
        "haystack_integrations.components.classifiers.typesafe.document_classifier.TypeSafeDocumentClassifier",
        "haystack_integrations.components.classifiers.nace.document_classifier.NaceDocumentClassifier",
    ),
    ("- TYPESAFE_API_KEY", "- NACE_API_KEY"),
    ("model: jev-latest", "model: drex-max-v1.1"),
]

EMAIL = "I was charged twice, please refund me today."

# Answers in the shape Jev's README documents for its classifier example
ANSWERS = {
    "department": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.95,
        "probabilities": {"billing": 0.97, "technical": 0.02, "sales": 0.01},
    },
    "urgency": {
        "type": "score",
        "score": 1.95,
        "confidence": 0.94,
        "legend": {"0": "can wait", "1": "this week", "2": "today"},
        "probabilities": {"0": 0.01, "1": 0.03, "2": 0.96},
    },
    "refund": {"type": "noul", "noul": 0.95},
}
ROUTE = {"route": {"type": "choice", "choice": "billing", "confidence": 0.93, "probabilities": {"billing": 0.93}}}


def _fake_drex(**call) -> DrexResponse:
    answers = ROUTE if "route" in call["questions"] else {key: ANSWERS[key] for key in call["questions"]}
    return DrexResponse.model_validate({"model": call["model"], "answers": answers, "usage": {"input_tokens": 9}})


def _mock(component) -> MagicMock:
    component._client = MagicMock()
    component._client.system_one.side_effect = _fake_drex
    return component._client


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)


def test_yaml_diff_is_only_the_documented_edits():
    jev = (FIXTURES / "typesafe_pipeline.yaml").read_text()
    for old, new in DOCUMENTED_YAML_EDITS:
        assert old in jev
        jev = jev.replace(old, new)
    assert jev == (FIXTURES / "nace_pipeline.yaml").read_text()


def test_switched_yaml_loads_and_runs():
    pipeline = Pipeline.loads((FIXTURES / "nace_pipeline.yaml").read_text())
    router_client = _mock(pipeline.get_component("router"))
    classifier_client = _mock(pipeline.get_component("classifier"))

    result = pipeline.run({"router": {"text": EMAIL}, "classifier": {"documents": [Document(content=EMAIL)]}})

    assert result["router"] == {"billing": EMAIL}
    # The saved YAML keeps `metadata_field: typesafe`, so filters that read Jev's answers keep working
    assert result["classifier"]["documents"][0].meta["typesafe"] == ANSWERS
    assert router_client.system_one.call_args.kwargs["model"] == "drex-max-v1.1"
    assert classifier_client.system_one.call_args.kwargs["model"] == "drex-max-v1.1"


def jev_router_example() -> dict:
    # Jev's TypeSafeTextRouter docstring example, with only the import and class name changed
    router = NaceTextRouter(
        labels={
            "billing": "Payments, invoices, refunds and charges",
            "technical": "Bugs, crashes and errors in the product",
        },
        instructions="Which team should handle this support request?",
        min_confidence=0.6,
    )
    _mock(router)
    return router.run(text="I was charged twice for my subscription.")


def jev_classifier_example() -> dict:
    # Jev's TypeSafeDocumentClassifier README example, with only the import and class name changed
    classifier = NaceDocumentClassifier(
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {"billing": None, "technical": None, "sales": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"],
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        },
    )
    _mock(classifier)
    return classifier.run(documents=[Document(content=EMAIL)])


def test_jev_router_example_runs_unchanged():
    assert jev_router_example() == {"billing": "I was charged twice for my subscription."}


def test_jev_classifier_example_runs_unchanged():
    result = jev_classifier_example()
    assert result["failed_documents"] == []
    answers = result["documents"][0].meta["nace"]
    # The same answer keys Jev documents for each question type
    assert set(answers["department"]) == {"type", "choice", "confidence", "probabilities"}
    assert set(answers["urgency"]) == {"type", "score", "confidence", "legend", "probabilities"}
    assert set(answers["refund"]) == {"type", "noul"}
