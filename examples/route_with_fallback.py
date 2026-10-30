# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""Route a support request through a complete Haystack pipeline.

From the repository root, with NACE_API_KEY set:
    python examples/route_with_fallback.py

Each input makes one billed Drex request. The handlers return handoff records;
replace them with your ticketing system or human-review queue in an application.
"""

from typing import Any

from haystack import Pipeline, component

from haystack_integrations.components.routers.nace import NaceTextRouter


@component
class TicketHandler:
    """Return a handoff record for the selected queue without sending a ticket."""

    def __init__(self, queue: str) -> None:
        self.queue = queue

    @component.output_types(ticket=dict[str, str])
    def run(self, text: str) -> dict[str, dict[str, str]]:
        return {"ticket": {"queue": self.queue, "text": text}}


def build_pipeline() -> Pipeline:
    """Connect each Drex output to its downstream handler, including human review."""
    pipeline = Pipeline()
    pipeline.add_component(
        "router",
        NaceTextRouter(
            model="drex-v2.0",
            labels={
                "billing": "Payments, invoices, refunds and charges",
                "technical": "Bugs, crashes and errors in the product",
            },
            instructions="Which team should handle this support request?",
            min_confidence=0.6,
            include_probabilities=True,
        ),
    )
    for output, queue in [
        ("billing", "billing"),
        ("technical", "technical"),
        ("low_confidence", "human_review"),
    ]:
        pipeline.add_component(queue, TicketHandler(queue))
        pipeline.connect(f"router.{output}", f"{queue}.text")
    return pipeline


def route_ticket(pipeline: Pipeline, text: str) -> dict[str, Any]:
    """Return the selected handler's ticket and Drex's probabilities."""
    result = pipeline.run({"router": {"text": text}})
    ticket = next(result[queue]["ticket"] for queue in ("billing", "technical", "human_review") if queue in result)
    return {"ticket": ticket, "decision": result["router"]["probabilities"]}


def main() -> None:
    pipeline = build_pipeline()
    for text in ["I was charged twice for my subscription.", "Can someone help me?"]:
        result = route_ticket(pipeline, text)
        print(result["ticket"])
        print(result["decision"])


if __name__ == "__main__":
    main()
