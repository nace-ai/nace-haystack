# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""Classify and route support emails with Drex in Haystack pipelines.

Run it with a Nace API key (synthetic emails only, a few cents of usage at most):

    NACE_API_KEY=nace_sk_... python examples/support_email_pipeline.py

It shows three things:
1. Indexing: `NaceDocumentClassifier` answers three questions about every email (department, urgency, refund) and
   the answers are stored with each document, ready for metadata filters.
2. Routing: `NaceTextRouter` sends each email to a team, with a `low_confidence` fallback. The email with a scanned
   invoice is routed together with the image on Drex Max; the scan is phone-photo size, so it is shrunk to fit one
   request and a warning says so. Pass `size=(2048, 2048)` to `ImageContent.from_file_path` to skip that.
3. Evaluation: `NaceFaithfulnessEvaluator` scores two draft replies against the refund policy.

Switching from Jev (typesafe-haystack)? Change the imports and class names, and the API key:

    -from haystack_integrations.components.routers.typesafe import TypeSafeTextRouter
    +from haystack_integrations.components.routers.nace import NaceTextRouter
    -export TYPESAFE_API_KEY=...
    +export NACE_API_KEY=...
"""

import json
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore
from make_invoice import make_invoice

from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier
from haystack_integrations.components.evaluators.nace import NaceFaithfulnessEvaluator
from haystack_integrations.components.routers.nace import NaceTextRouter

DATA = Path(__file__).parent / "data"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "Invoices, payments, refunds and taxes",
            "technical": "Crashes, errors, sign-in problems and bugs",
            "sales": "Pricing, plans, demos and new purchases",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this email?",
        "criteria": ["can wait", "this week", "today"],
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}

TEAMS = {
    "billing": "Invoices, payments, refunds and taxes",
    "technical": "Crashes, errors, sign-in problems and bugs",
    "sales": "Pricing, plans, demos and new purchases",
}

REFUND_POLICY = (
    "Duplicate charges are refunded in full within 5 business days. "
    "Annual plans cancelled within 30 days of renewal get a prorated refund. Other fees are not refundable."
)


def load_emails() -> list[dict]:
    return json.loads((DATA / "emails.json").read_text())


def scanned_invoice() -> Path:
    path = DATA / "scanned_invoice.png"
    if not path.exists():
        make_invoice().save(path)
    return path


def index_emails(emails: list[dict]) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    indexing = Pipeline()
    indexing.add_component("classifier", NaceDocumentClassifier(questions=QUESTIONS))
    indexing.add_component("writer", DocumentWriter(document_store=store))
    indexing.connect("classifier.documents", "writer.documents")
    documents = [
        Document(content=f"Subject: {email['subject']}\n\n{email['body']}", meta={"email_id": email["id"]})
        for email in emails
    ]
    indexing.run({"classifier": {"documents": documents}})
    return store


def main() -> None:
    emails = load_emails()

    print("1. Indexing: every email classified once, answers stored with the document\n")
    store = index_emails(emails)
    for document in sorted(store.filter_documents(), key=lambda document: document.meta["email_id"]):
        answers = document.meta["nace"]
        department = answers["department"]
        print(
            f"  {document.meta['email_id']}  {department['choice']:<9} ({department['confidence']:.2f})  "
            f"urgency {answers['urgency']['score']:.1f}/2  refund {answers['refund']['noul']:.2f}"
        )
    refund_requests = store.filter_documents(
        filters={"field": "meta.nace.refund.noul", "operator": ">", "value": 0.5},
    )
    print(f"\n  Refund requests found with a metadata filter: {sorted(d.meta['email_id'] for d in refund_requests)}")

    print("\n2. Routing: each email to a team, the scanned invoice read on Drex Max\n")
    router = NaceTextRouter(
        labels=TEAMS,
        instructions="Which team should handle this email? Image 1, if present, is the customer's attachment.",
        min_confidence=0.6,
        include_probabilities=True,
    )
    for email in emails:
        images = [ImageContent.from_file_path(scanned_invoice())] if email["attachment"] else None
        routed = router.run(text=f"Subject: {email['subject']}\n\n{email['body']}", images=images)
        team = next(output for output in routed if output != "probabilities")
        answer = routed["probabilities"]
        note = " (with the scanned invoice)" if images else ""
        print(f"  {email['id']}  -> {team:<14} confidence {answer['confidence']:.2f}{note}")

    invoice_reader = NaceTextRouter(
        labels={"total_4212": "$4,212.00", "total_2106": "$2,106.00", "total_1042": "$1,042.00"},
        instructions="What total amount is due on the invoice in Image 1?",
        include_probabilities=True,
    )
    total = invoice_reader.run(
        text="The customer's attachment.", images=[ImageContent.from_file_path(scanned_invoice())]
    )
    answer = total["probabilities"]
    amount = invoice_reader.labels[answer["choice"]]
    print(f"\n  Total read from the scan: {amount} (confidence {answer['confidence']:.2f})")

    print("\n3. Evaluation: two draft replies to e1 checked against the refund policy\n")
    evaluator = NaceFaithfulnessEvaluator()
    result = evaluator.run(
        questions=[
            "What happens to the duplicate charge on INV-1042?",
            "What happens to the duplicate charge on INV-1042?",
        ],
        contexts=[[REFUND_POLICY], [REFUND_POLICY]],
        predicted_answers=[
            "The duplicate charge will be refunded in full within 5 business days.",
            "We can offer a 50% credit on your next invoice instead of a refund.",
        ],
    )
    for name, score in zip(["policy-backed reply", "made-up offer"], result["individual_scores"], strict=True):
        print(f"  {name:<20} faithfulness {score:.2f}")


if __name__ == "__main__":
    main()
