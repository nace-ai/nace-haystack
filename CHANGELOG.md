# Changelog

## [Unreleased]

## [0.1.1] - 2026-10-30

### Fixed

- Stop async batch workers from sending queued requests after a terminal failure.
- Treat missing required answers and invalid probabilities as response-validation failures, respecting per-item failure handling.
- Cap formatted API messages, including structured issues, while preserving recovery guidance.
- Bind the default Nace credential to Nace's HTTPS production origins. Custom providers now require an explicit
  provider key; saved pipelines pin their destination, empty keys are refused, and OpenRouter host matching is exact.
- Strip ambient Vercel bypass headers on other providers and refuse cross-origin redirects in sync and async clients.
- Check all completed batch requests for failures before starting replacements. Requests already in flight may finish.
- Restore deepset copyright notices in components and tests adapted from its TypeSafe integration, with source attribution.
- Point insufficient-credit errors to the working `/dashboard/billing` page.

## [0.1.0] - 2026-10-29

First release.

### Added

- `NaceTextRouter`: routes a text to one output per label with Drex, with a `low_confidence` fallback and an opt-in
  `probabilities` output. Takes images (`ImageContent`, `ByteStream`, or files uploaded to Drex).
- `NaceDocumentClassifier`: answers typed questions about each document and stores Drex's answers in its metadata,
  with parallel requests and per-document failure handling.
- `NaceFaithfulnessEvaluator` and `NaceContextRelevanceEvaluator`: the inputs and outputs of Haystack's built-in LLM
  evaluators, with one Drex request per item.
- Same parameters and outputs as deepset's `typesafe-haystack` components, so Jev pipelines switch with the documented
  edits.
- Default model `drex-max-v1.1` (Drex Max v1.1). Any model ID is sent as given, and answers keep every field Drex returns, so new Drex
  models, question types and answer fields work without a new release.
- Images too big for one request are shrunk to fit (Pillow, optional).
- OpenRouter backend, attribution headers, a 60-second timeout, and retries on 429 and 529 that honour
  `retry-after-ms`.
- Readable errors that keep nace-sdk's error classes.

### Changed

- Licensed under Apache-2.0, the licence of Haystack's own integrations.

### Fixed

Found while preparing the release, before anything was published:

- With `raise_on_failure=True`, the classifier and evaluators stop sending requests as soon as one fails, instead of
  sending every queued request first.
- Waits that the server asks for between retries are capped at 60 seconds.
- OpenRouter variants of text-only models, such as `nace-ai/drex-v2.0:floor`, are refused images before sending,
  like the models themselves.
- A `ByteStream` typed `application/octet-stream` is sent with its real image type.
- A 402 from OpenRouter points to OpenRouter's credits page instead of Nace billing.
- HTML error pages are summarised in one line, and long error messages are cut to 500 characters.
- `run_async` works again after the event loop changes, for example a second `asyncio.run` or notebook cell; it
  failed with "Event loop is closed".
- A 16-bit grayscale scan that has to be shrunk keeps its content instead of turning into a blank white page.
- A 404 or 405 error names the URL it went to, so a wrong base URL is easy to spot.
- The router raises a clear error when the backend answers with a label it wasn't given, instead of dropping the
  text.
- A label named `type` with a description is refused, because a saved pipeline with it couldn't be loaded.
- The evaluators explain that `images` takes one list per item when given a flat list.
- `pydantic>=2.0` and `filetype>=1.0.0` are declared as dependencies; `filetype` 0.x can't be imported on
  Python 3.11.
