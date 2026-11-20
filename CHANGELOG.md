# Changelog

## [Unreleased]

### Added

- `NaceDocumentConverter`: converts file paths, `ByteStream` sources or `https://` URL sources into Markdown
  Documents with NDI Parse, optionally one Document per page or section, with parallel sources and per-source
  failure handling. Uploaded files and jobs are deleted after each run; a result NDI cut at 1 MiB is never returned
  as the full content. A succeeded result without the Markdown, without the chunks a `chunking` asked for, or with
  a chunk that is not an object with content, raises instead of returning an empty Document or dropping the source.
- `NaceDocumentExtractor`: fills a JSON Schema from each source with NDI Extract, as one Document per source with
  the extracted data as JSON in `content` and NDI's field report in `meta["extraction"]`. A succeeded result
  without `data` raises instead of producing a Document of `null`.
- `NaceWorkspaceSearch`: answers one question over a Nace workspace's ingested files with Drex Search, as a
  `GeneratedAnswer` whose documents are the cited evidence quotes, without scores. An agent answer that lists
  documents keeps its catalog rows in the answer's `meta["catalog"]`, with `meta["exhausted"]` when the agent ran
  out of steps and the answer is partial. A succeeded search without an answer output raises instead of returning
  an empty answer.
- The wait for an NDI job is bounded by `job_timeout`: the create's hold and request timeout, and every job
  read's hold and HTTP timeout, are capped to the budget left where the SDK allows (the SDK enforces a minimum create hold and HTTP timeout, so with a very small budget the first request can outlast it). The bound is best-effort: a request already in
  flight, an SDK retry, or the cleanup after a timeout can run slightly past the deadline.
- The NDI components never expose the API key value in an error or log, also when a gateway echoes the
  `Authorization` value back, and the metadata of a URL source keeps only its scheme, host and path, so the
  signature of a presigned URL never reaches a serialized Document. They refuse a `job_timeout` that is not a
  finite number greater than 0.
- A create that NDI answers with `503 job_pending` can stand for an accepted job whose id is not known. The raised
  error keeps the create's idempotency key, so the job can be resumed with `nace-sdk` and then deleted; everything
  whose id is known is still cleaned up. A create whose response is lost (a timeout after NDI accepted it) can
  equally leave a job running: `nace-sdk`'s timeout errors carry no key, and such a job is found with
  `client.jobs.list()`.

### Changed

- NDI needs `nace-sdk>=0.9.0`, up from `>=0.3.0`.
- The request hook allows a request to another origin only when it carries no credentials and is a clean HTTPS URL:
  NDI uploads and signed-link downloads use such requests. A request with `Authorization` to another origin, or one
  over plain HTTP, is still refused.

## [0.1.2] - 2026-10-30

### Added

- Nace and Haystack logos at the top of the README, and CI and license badges under its title. The README is
  also the PyPI project page, so they now show there too.

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
