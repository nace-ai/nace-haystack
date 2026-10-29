# nace-haystack

Haystack components for [Drex](https://console.nace.ai/docs), Nace's decision models.

Drex models don't generate text. They answer typed questions about a text, JSON or images in one request and return
calibrated probabilities:

- `choice` picks one label from a set of options,
- `score` rates the input on ordered levels,
- `noul` gives the probability that a yes/no statement is true.

This package brings them to Haystack pipelines:

| Component | What it does |
| --- | --- |
| `NaceTextRouter` | Routes a text (and optional images) to one output per label, with a `low_confidence` fallback |
| `NaceDocumentClassifier` | Answers questions about each document and stores the answers in its metadata |
| `NaceFaithfulnessEvaluator` | Judges whether answers are supported by their contexts, like Haystack's `FaithfulnessEvaluator` |
| `NaceContextRelevanceEvaluator` | Judges whether retrieved contexts help answer the question, like `ContextRelevanceEvaluator` |

The router and classifier mirror deepset's `typesafe-haystack` components for Jev, so a Jev pipeline switches to Drex
with a few edits (see [Switching from TypeSafe](#switching-from-typesafe)).

## Installation

```bash
pip install nace-haystack
```

Create an API key at [console.nace.ai](https://console.nace.ai) and set it as `NACE_API_KEY`.

## Usage

### Route text

```python
from haystack_integrations.components.routers.nace import NaceTextRouter

router = NaceTextRouter(
    labels={
        "billing": "Payments, invoices, refunds and charges",
        "technical": "Bugs, crashes and errors in the product",
    },
    instructions="Which team should handle this support request?",
    min_confidence=0.6,
)

print(router.run(text="I was charged twice for my subscription."))
# {'billing': 'I was charged twice for my subscription.'}
```

Texts whose answer is less confident than `min_confidence` go to the `low_confidence` output, for a fallback such
as an LLM or a person. With `include_probabilities=True`, a `probabilities` output carries the full answer, so
thresholds stay in your pipeline.

### Classify documents

```python
from haystack import Document

from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier

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

result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
print(result["documents"][0].meta["nace"]["department"])
# {'type': 'choice', 'choice': 'billing', 'confidence': 0.97, 'probabilities': {...}}
```

Each document is one request carrying every question, with up to `max_workers` requests (default 3) at once.
Documents that can't be classified are returned in `failed_documents` with the reason in
`meta["classification_error"]`.

### Evaluate RAG answers

```python
from haystack_integrations.components.evaluators.nace import NaceFaithfulnessEvaluator

evaluator = NaceFaithfulnessEvaluator()
result = evaluator.run(
    questions=["Who won the race?"],
    contexts=[["Ada won the 2026 city race."]],
    predicted_answers=["Ada won the race."],
)
print(result["score"], result["individual_scores"])
```

Both evaluators take the same inputs and return the same outputs (`score`, `individual_scores`, `results`, `meta`) as
Haystack's built-in LLM evaluators. Each item is one Drex request, and its score is the probability that the answer
is faithful (or that the contexts are relevant).

## Models

The default model is `drex-max-v1.1` (Drex Max v1.1), which reads text, JSON and images. The default moves to the
newest Drex model in package releases, and a new model can shift probabilities, so **pin `model=` in production** if
your pipeline relies on thresholds.

`drex-v2.0` reads text only and bills fewer tokens on short texts: for a short email with three questions, we
measured 74 input tokens on `drex-v2.0` and 165 on `drex-max-v1.1` (2026-10-29). See
[Drex's limits](https://console.nace.ai/docs/drex/reference/limits) for questions per request and state sizes.
Any model ID is sent as given, so new Drex models work without a new release of this package.

## Images

The router and the evaluators take images: `ImageContent`, `ByteStream`, or files uploaded to Drex with nace-sdk
(`client.documents.upload(...)`). Prompts refer to them as "Image 1", "Image 2" and so on.

```python
from haystack.dataclasses import ImageContent

router = NaceTextRouter(
    labels={"invoice": "A bill or invoice", "receipt": "A proof of payment", "other": None},
    instructions="What kind of document is Image 1?",
)
router.run(text="A customer attached a scan.", images=[ImageContent.from_file_path("scan.jpg")])
```

Images too big to send in one request are shrunk automatically, with a logged warning. Drex Max reads images at about
1.6 megapixels (1,638,400 pixels), and an image is never shrunk below about 1 megapixel: if the text leaves too little
room for the images, you get a clear error instead of a blurry image. Photos are turned upright using their EXIF
orientation before shrinking. Shrinking needs `pip install pillow`; to skip it, downscale earlier with
`ImageFileToImageContent(size=(2048, 2048))` or `PDFToImageContent(size=(2048, 2048))`. PDFs must be converted to
images first.

## Other backends

Set `api_base_url` to use Drex through [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`, with
model IDs such as `nace-ai/drex-max`) or another TypeSafe-compatible host. OpenRouter takes inline images only.

## Errors and retries

Requests that hit a rate limit (429) or an overloaded model (529) are retried, waiting as long as Drex asks. Other
errors are raised as nace-sdk's error classes with a readable message, for example:

```text
422 from Drex: the request is invalid
  questions.department.criteria: must have at least 2 labels
```

## Switching from TypeSafe

The router and classifier take the same parameters and produce the same outputs as `TypeSafeTextRouter` and
`TypeSafeDocumentClassifier`.

In Python code that uses Jev's default model, change the import, the class name and the API key:

```diff
-from haystack_integrations.components.routers.typesafe import TypeSafeTextRouter
+from haystack_integrations.components.routers.nace import NaceTextRouter

-router = TypeSafeTextRouter(labels=..., min_confidence=0.6)
+router = NaceTextRouter(labels=..., min_confidence=0.6)
```

```diff
-export TYPESAFE_API_KEY=...
+export NACE_API_KEY=...
```

The classifier writes answers to `meta["nace"]` by default. If your code reads `meta["typesafe"]` (in a filter or a
`MetadataRouter` rule, for example), pass `metadata_field="typesafe"` to keep the old key.

Questions built with `typesafe_sdk` objects (`Choice`, `Score`, `Noul`) work as they are; you can switch them to
`nace_sdk`'s classes of the same names when convenient. Where your code catches `TypeSafeError`, catch
`nace_sdk.NaceError` instead.

In a saved YAML pipeline, make three edits per component:

```diff
-    type: haystack_integrations.components.routers.typesafe.text_router.TypeSafeTextRouter
+    type: haystack_integrations.components.routers.nace.text_router.NaceTextRouter
...
-        - TYPESAFE_API_KEY
+        - NACE_API_KEY
...
-      model: jev-latest
+      model: drex-max-v1.1
```

The classifier's type becomes `haystack_integrations.components.classifiers.nace.document_classifier.NaceDocumentClassifier`.
A saved pipeline keeps `metadata_field: typesafe`, so answers stay where your filters read them. Model names aren't
translated: Drex rejects `jev-latest` with a message listing its models.

Drex's own guide covers the API side: [Migrate from TypeSafe](https://console.nace.ai/docs/drex/guides/migrate-from-typesafe).

## License

Apache-2.0
