# Releasing nace-haystack

## On every Drex model release

1. Bump `DEFAULT_MODEL` in `src/haystack_integrations/common/nace/client.py` to the new model.
2. Add the new model to `TEXT_ONLY_MODELS` in the same file only if it reads text only.
3. Run the live tests: `NACE_API_KEY=... hatch run test:integration`.
4. Add a CHANGELOG entry that says the default model changed and that probabilities may shift, so users who rely on thresholds should pin `model=`.
5. If Drex's request size limit changed, update `MAX_REQUEST_BYTES` in `src/haystack_integrations/common/nace/images.py`.
6. Release (next section), then do the same for langchain-nace and llama-index-nace.

## Releasing a version

Releases are published by `.github/workflows/release.yml` when a `v*` tag is pushed, through PyPI Trusted Publishing
(no PyPI token is stored anywhere). This is the same setup as nace-sdk's release.

1. Move the entries under `[Unreleased]` in `CHANGELOG.md` to a heading for the new version, and merge that to `main`.
2. Rehearse: run the Release workflow by hand (Actions → Release → Run workflow). It runs every check and builds the
   package, but doesn't publish.
3. Tag `vX.Y.Z` on `main` and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`. hatch-vcs reads the
   package version from the tag (`tag-pattern` in `pyproject.toml`), so no file holds the version. The workflow
   refuses to publish if the built version doesn't match the tag.
4. A version can never be uploaded to PyPI twice. If the publish fails after the upload, don't retag: fix forward
   with the next patch version.
5. Check the release: `pip install nace-haystack==X.Y.Z` in a clean environment and run
   `examples/support_email_pipeline.py`.
6. Update the Haystack integrations listing if the API changed. The page is `listing/nace.md` here and
   `integrations/nace.md` in `deepset-ai/haystack-integrations`.

### One-time PyPI setup

The project trusts this repository's workflow, configured on pypi.org under Publishing:

- project `nace-haystack`;
- owner `nace-ai`;
- repository `nace-haystack`;
- workflow `release.yml`;
- environment `pypi`.

The repository needs a GitHub environment named `pypi`.
