# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import nace_sdk
from haystack import Document, component, default_to_dict, logging
from haystack.dataclasses import GeneratedAnswer
from haystack.utils import Secret

from haystack_integrations.common.nace.client import bind_credentials, client_options, make_client
from haystack_integrations.common.nace.documents import finished_job
from haystack_integrations.common.nace.errors import readable_error

logger = logging.getLogger(__name__)


@component
class NaceWorkspaceSearch:
    """
    Answers one question over the ingested files of a Nace workspace, with Drex Search.

    The workspace must already exist and its files must be ingested before this component runs: upload and ingest
    them in the Nace console, or with nace-sdk (`client.documents.upload(..., workspace_id=...)`, then
    `client.workspaces.ingest`). NDI searches the files of that workspace only, not the web.

    This component is not a retriever: it returns no scores. It returns one `GeneratedAnswer` whose `documents` are
    the evidence quotes the answer cites, each with `source_path`, `source_file_id` and, when NDI names a page,
    `page_number` in its metadata. The search job is deleted after every run.

    ### Usage example

    ```python
    from haystack_integrations.components.connectors.nace import NaceWorkspaceSearch

    search = NaceWorkspaceSearch(workspace_id="11111111-1111-1111-1111-111111111111")
    result = search.run(query="What is the total of invoice INV-1042?")
    print(result["answers"][0].data)
    # 'The total of invoice INV-1042 is $284.00.'
    ```
    """

    def __init__(
        self,
        workspace_id: str,
        *,
        mode: Literal["auto", "fact", "agent"] = "auto",
        path_prefix: str | None = None,
        api_key: Secret | None = None,
        api_base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        job_timeout: float = 600.0,
    ) -> None:
        """
        Creates a NaceWorkspaceSearch.

        :param workspace_id:
            The id of the workspace to search.
        :param mode:
            How the question is answered: `"auto"` picks fact search or the search agent, `"fact"` and `"agent"`
            pick one of them.
        :param path_prefix:
            An optional prefix that narrows the search to the workspace files under it.
        :param api_key:
            A provider key. Defaults to `NACE_API_KEY` on Nace production origins; required for custom hosts.
        :param api_base_url:
            Base URL of the API. If `None`, the `NACE_BASE_URL` environment variable is used, or
            `https://console.nace.ai` if it is unset.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, 60 seconds are used.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param job_timeout:
            How long to wait for the search job to finish, in seconds.
        """
        self.workspace_id = workspace_id
        self.mode = mode
        self.path_prefix = path_prefix
        self.api_key, self.api_base_url = bind_credentials(api_key, api_base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.job_timeout = job_timeout
        self._client: nace_sdk.NaceClient | None = None

    def _client_options(self) -> dict[str, Any]:
        return client_options(self.api_key, self.api_base_url, self.timeout, self.max_retries)

    def warm_up(self) -> None:
        """
        Creates the synchronous Nace client.
        """
        if self._client is None:
            self._client = make_client(self._client_options())

    def close(self) -> None:
        """
        Releases the synchronous Nace client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            workspace_id=self.workspace_id,
            mode=self.mode,
            path_prefix=self.path_prefix,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            job_timeout=self.job_timeout,
        )

    def _evidence_documents(self, evidences: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for evidence in evidences:
            meta = {
                "source_path": evidence.get("source_path"),
                "source_file_id": evidence.get("source_file_id"),
            }
            page = evidence.get("page")
            if page is not None:
                meta["page_number"] = page
            documents.append(Document(content=evidence.get("quote") or "", meta=meta))
        return documents

    @component.output_types(answers=list[GeneratedAnswer])
    def run(self, query: str) -> dict[str, list[GeneratedAnswer]]:
        """
        Answers one question over the workspace's ingested files with Drex Search.

        :param query:
            The question to answer.
        :returns:
            A dictionary with the following keys:
            - `answers`: One `GeneratedAnswer`: the answer, the query it answers, and one Document per evidence
              quote, without scores.
        :raises nace_sdk.NaceError:
            If the search or the wait for it fails, or the job did not succeed or has no result.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        job_id: str | None = None
        try:
            job = client.search.automatic(
                self.workspace_id,
                query=query,
                mode=self.mode,
                path_prefix=self.path_prefix,
                wait_seconds=60,
            )
            job_id = job.job_id
            job = finished_job(client, job, self.job_timeout)
            output = (job.result or {}).get("output") or {}
            # An agent-routed search answers with the answer, its confidence and its evidences inside `output.answer`.
            body = output["answer"] if isinstance(output.get("answer"), dict) else output
            answer = GeneratedAnswer(
                data=body.get("answer") or "",
                query=query,
                documents=self._evidence_documents(body.get("evidences") or []),
                meta={"confidence": body.get("confidence")},
            )
            return {"answers": [answer]}
        except nace_sdk.NaceError as error:
            raise readable_error(error, "Nace") from error
        finally:
            if job_id is not None:
                try:
                    client.jobs.delete(job_id)
                except nace_sdk.NaceError as error:
                    logger.warning("Could not delete search job {job_id}: {error}", job_id=job_id, error=error)
