# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component
from haystack.utils import Secret

from haystack_integrations.common.nace.client import DEFAULT_MODEL
from haystack_integrations.common.nace.images import ImageInput

from ._base import _NaceEvaluator

CONTEXT_RELEVANCE_INSTRUCTIONS = "Do the contexts contain information that helps answer the question?"


@component
class NaceContextRelevanceEvaluator(_NaceEvaluator):
    """
    Judges whether the retrieved contexts help answer each question, with a Drex decision model.

    A drop-in for Haystack's `ContextRelevanceEvaluator`: same inputs and outputs, but each item is one Drex request
    that returns the probability that the contexts help answer the question.

    ### Usage example

    ```python
    from haystack_integrations.components.evaluators.nace import NaceContextRelevanceEvaluator

    evaluator = NaceContextRelevanceEvaluator()
    result = evaluator.run(questions=["Who won the race?"], contexts=[["Ada won the 2026 city race."]])
    print(result["score"])
    # 0.98
    ```
    """

    def __init__(
        self,
        *,
        instructions: str = CONTEXT_RELEVANCE_INSTRUCTIONS,
        model: str = DEFAULT_MODEL,
        api_key: Secret | None = None,
        api_base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Creates a NaceContextRelevanceEvaluator.

        :param instructions:
            The yes/no question Drex answers about each item.
        :param model:
            The Drex model to use. Defaults to the newest Drex model; pin it in production.
        :param api_key:
            A provider key. Defaults to `NACE_API_KEY` on Nace production origins; required for custom hosts.
        :param api_base_url:
            Base URL of the API. If `None`, the `NACE_BASE_URL` environment variable is used, or
            `https://console.nace.ai` if it is unset.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, 60 seconds are used.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param max_workers:
            Maximum number of requests running at once. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed request raises its error. If `False`, a failed item scores NaN, is marked
            with `status: "error"` in `results`, and is left out of the mean `score`.
        """
        # @component rebuilds the class, so zero-argument super() would point at the undecorated one
        _NaceEvaluator.__init__(
            self,
            instructions=instructions,
            model=model,
            api_key=api_key,
            api_base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_workers=max_workers,
            raise_on_failure=raise_on_failure,
        )

    def _states(
        self, questions: list[str], contexts: list[list[str]], images: list[list[ImageInput]] | None
    ) -> list[dict[str, Any]]:
        self._check_contexts(contexts)
        self._check_images(images)
        self._check_lengths({"questions": questions, "contexts": contexts, "images": images})
        return [
            {"question": question, "contexts": list(context)}
            for question, context in zip(questions, contexts, strict=True)
        ]

    @component.output_types(
        score=float, individual_scores=list[float], results=list[dict[str, Any]], meta=list[dict[str, Any]] | None
    )
    def run(
        self, questions: list[str], contexts: list[list[str]], images: list[list[ImageInput]] | None = None
    ) -> dict[str, Any]:
        """
        Judges whether each question's contexts help answer it.

        :param questions:
            The questions.
        :param contexts:
            The contexts retrieved for each question.
        :param images:
            Optional images for each item, which Drex Max reads as "Image 1", "Image 2" and so on.
        :returns:
            A dictionary with `score`, `individual_scores`, `results` and `meta`, as in Haystack's
            `ContextRelevanceEvaluator`; failed items score NaN when `raise_on_failure` is `False`.
        :raises ValueError:
            If the inputs have different lengths, or images are passed to a model that reads text only.
        """
        return self._evaluate(self._states(questions, contexts, images), images)

    @component.output_types(
        score=float, individual_scores=list[float], results=list[dict[str, Any]], meta=list[dict[str, Any]] | None
    )
    async def run_async(
        self, questions: list[str], contexts: list[list[str]], images: list[list[ImageInput]] | None = None
    ) -> dict[str, Any]:
        """
        Asynchronously judges whether each question's contexts help answer it.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param questions:
            The questions.
        :param contexts:
            The contexts retrieved for each question.
        :param images:
            Optional images for each item.
        :returns:
            The same dictionary as `run`.
        """
        return await self._evaluate_async(self._states(questions, contexts, images), images)
