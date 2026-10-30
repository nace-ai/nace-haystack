# nace-haystack

[Drex](https://console.nace.ai/docs) decision-model components for
[Haystack](https://github.com/deepset-ai/haystack), deepset's open-source Python framework for AI pipelines,
agents, and retrieval-augmented generation (RAG). Maintained by [Nace AI](https://nace.ai).

Haystack connects components for tasks such as retrieving documents, calling models, and routing results. This
package adds Drex decisions and evaluators to those pipelines. Start with the
[Haystack documentation](https://docs.haystack.deepset.ai/docs/intro) if you are new to the framework.

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

The router and classifier mirror deepset's
[`typesafe-haystack` integration](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/typesafe)
for Jev, so a Jev pipeline switches to Drex
with a few edits (see [Switching from TypeSafe](#switching-from-typesafe)).

## Installation

Python 3.11 or later is required. Install the published package in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install nace-haystack
```

Create an API key at [console.nace.ai](https://console.nace.ai), then set it in your shell:

```bash
export NACE_API_KEY="your-nace-api-key"
# Windows PowerShell: $env:NACE_API_KEY="your-nace-api-key"
```

The examples below call the hosted Drex API and consume your account's credit. Keep the key out of source code and
notebook outputs. See [pricing](https://console.nace.ai/docs/drex/reference/pricing) for current rates.

## Usage

### Quickstart: route one request

Save this snippet as `route_text.py` and run `python route_text.py` in the environment above. This text-only example
pins `drex-v2.0`; use `drex-max-v1.1` when you need images.

```python
from haystack_integrations.components.routers.nace import NaceTextRouter

router = NaceTextRouter(
    model="drex-v2.0",
    labels={
        "billing": "Payments, invoices, refunds and charges",
        "technical": "Bugs, crashes and errors in the product",
    },
    instructions="Which team should handle this support request?",
    min_confidence=0.6,
    include_probabilities=True,
)

print(router.run(text="I was charged twice for my subscription."))
# Illustrative output; probabilities vary:
# {'billing': 'I was charged twice for my subscription.',
#  'probabilities': {'type': 'choice', 'choice': 'billing', 'confidence': 0.97,
#                    'probabilities': {'billing': 0.97, 'technical': 0.03}}}
```

Texts whose answer is less confident than `min_confidence` go to the `low_confidence` output, for a fallback such
as an LLM or a person. With `include_probabilities=True`, a `probabilities` output carries the full answer, so
thresholds stay in your pipeline.

`choice` is the selected label, `confidence` is its probability, and `probabilities` contains every label's
probability. A result of 0.97 is not a guarantee that the decision is correct. The 0.6 threshold is an example:
choose your threshold using representative labeled inputs and the cost of a wrong route. Raise it when uncertain
requests should go to review more often. Low confidence uses the fallback output; API failures raise an error
rather than automatically using that output.

### Connect the decision to a complete pipeline

[Run the routing example](https://github.com/nace-ai/nace-haystack/blob/main/examples/route_with_fallback.py) to see
all three outputs connected to downstream handlers:

```text
support request → Drex router → billing handler
                             → technical handler
                             → human-review handler (below the confidence threshold)
```

The example uses `Pipeline.add_component` and `Pipeline.connect`. Only the selected handler runs; it returns a
`ticket` record naming its queue, alongside the original Drex probabilities. It does not send emails or create real
tickets. Replace the handlers with your own actions in an application. The ambiguous sample may take different
routes on different runs, depending on its confidence.

The scripts and notebook use files from this repository, which are not installed by `pip install nace-haystack`.
Clone the repo and run the example from its root:

```bash
git clone https://github.com/nace-ai/nace-haystack.git
cd nace-haystack
python examples/route_with_fallback.py
```

### Classify documents

```python
from haystack import Document

from haystack_integrations.components.classifiers.nace import NaceDocumentClassifier

classifier = NaceDocumentClassifier(
    model="drex-v2.0",
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

evaluator = NaceFaithfulnessEvaluator(model="drex-v2.0")
result = evaluator.run(
    questions=["Who won the race?"],
    contexts=[["Ada won the 2026 city race."]],
    predicted_answers=["Ada won the race."],
)
print(result["score"], result["individual_scores"])
```

Both evaluators take the same inputs and return the same outputs (`score`, `individual_scores`, `results`, `meta`) as
Haystack's built-in LLM evaluators. Each item is one Drex request, and its score is the probability that the answer
is faithful (or that the contexts are relevant). `score` is the mean over the successful items, while
`individual_scores` keeps one value per item. If you set `raise_on_failure=False`, a failed item is NaN and is
excluded from the mean; check `results` for errors before using an aggregate.

To evaluate retrieval before generating an answer:

```python
from haystack_integrations.components.evaluators.nace import NaceContextRelevanceEvaluator

evaluator = NaceContextRelevanceEvaluator(model="drex-v2.0")
result = evaluator.run(
    questions=["Who won the race?"],
    contexts=[["Ada won the 2026 city race."]],
)
print(result["score"], result["individual_scores"])
```

### Run the full support-email example and cookbook

The [support-email pipeline](https://github.com/nace-ai/nace-haystack/blob/main/examples/support_email_pipeline.py)
classifies eight synthetic emails into document metadata, filters refund requests, routes emails and an invoice
image, and checks draft replies against a refund policy. From the cloned repository root:

```bash
python -m pip install pillow
python examples/support_email_pipeline.py
```

Pillow draws the synthetic invoice and shrinks large images. The example generates its own invoice file, so you do
not need to upload a customer document.

The [cookbook notebook](https://github.com/nace-ai/nace-haystack/blob/main/notebooks/nace_haystack_cookbook.ipynb)
walks through the same workflow step by step. Launch Jupyter from the shell where `NACE_API_KEY` is set:

```bash
python -m pip install pillow jupyterlab
python -m jupyterlab notebooks/nace_haystack_cookbook.ipynb
```

Select the virtual environment's Python kernel. The notebook reads sample files from the checkout; run it from the
repository root or the `notebooks/` directory.

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

from haystack_integrations.components.routers.nace import NaceTextRouter

router = NaceTextRouter(
    model="drex-max-v1.1",
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

Set `api_base_url` and an explicit provider key to use Drex through [OpenRouter](https://openrouter.ai) or another
TypeSafe-compatible host. OpenRouter takes inline images only.

```python
from haystack.utils import Secret
from haystack_integrations.components.routers.nace import NaceTextRouter

router = NaceTextRouter(
    labels=["billing", "technical"],
    model="nace-ai/drex-max",
    api_base_url="https://openrouter.ai/api/v1",
    api_key=Secret.from_env_var("OPENROUTER_API_KEY"),
)
```

Set `OPENROUTER_API_KEY` in your environment before running the router. Use a separate provider environment
variable for other hosts; `NACE_API_KEY` is reserved for Nace's production API. Custom destinations require an
explicit `api_key`, and an empty key is refused before a request is sent. Ambient Vercel bypass headers are
removed from requests to other providers, and redirects to a different origin are refused.

The base URL is resolved when the component is constructed, then saved with the pipeline. Changing
`NACE_BASE_URL` later does not redirect an existing component. To change providers in a saved pipeline, update
both its base URL and its provider key reference. Older saved pipelines that send `NACE_API_KEY` to a custom
host must switch to a separate provider key reference before loading.

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

The router, document classifier and their tests are adapted from deepset's Apache-2.0-licensed
[`typesafe-haystack` integration](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/typesafe).
The original deepset copyright notices are retained alongside Nace's notices for its modifications.
