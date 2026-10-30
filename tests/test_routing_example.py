# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import httpx
import pytest

from haystack_integrations.common.nace.client import make_client
from haystack_integrations.components.routers.nace import text_router

EXAMPLE = Path(__file__).parents[1] / "examples" / "route_with_fallback.py"


@pytest.mark.skipif(
    not EXAMPLE.is_file(), reason="Example scripts are repository-only, excluded from source distributions."
)
@pytest.mark.parametrize(
    ("probabilities", "expected_queue"),
    [
        ({"billing": 0.97, "technical": 0.03}, "billing"),
        ({"billing": 0.1, "technical": 0.9}, "technical"),
        ({"billing": 0.55, "technical": 0.45}, "human_review"),
    ],
)
def test_routing_example_hands_off_to_the_selected_queue(monkeypatch, probabilities, expected_queue):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_example_test")
    monkeypatch.delenv("NACE_BASE_URL", raising=False)
    monkeypatch.delenv("DREX_BASE_URL", raising=False)
    seen = []
    choice = max(probabilities, key=probabilities.get)

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "drex-v2.0",
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": probabilities[choice],
                        "probabilities": probabilities,
                    },
                },
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(text_router, "make_client", lambda options: make_client({**options, "transport": transport}))
    example = runpy.run_path(str(EXAMPLE))
    pipeline = example["build_pipeline"]()
    try:
        result = example["route_ticket"](pipeline, "Please help with this request.")
    finally:
        pipeline.get_component("router").close()
    assert result["ticket"] == {"queue": expected_queue, "text": "Please help with this request."}
    assert result["decision"]["probabilities"] == probabilities
    assert len(seen) == 1
    assert seen[0]["model"] == "drex-v2.0"
    assert seen[0]["state"] == "Please help with this request."
