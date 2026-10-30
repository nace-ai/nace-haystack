# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import httpx
import nace_sdk
from pydantic import BaseModel, ConfigDict

from haystack_integrations.common.nace.client import is_text_only
from haystack_integrations.common.nace.images import MAX_REQUEST_BYTES, ImageInput, fit_images, to_wire_image

Question: TypeAlias = Mapping[str, Any] | nace_sdk.Choice | nace_sdk.Score | nace_sdk.Noul

# Room for the commas and quotes around each image in the request body.
_JSON_HEADROOM = 64 * 1024


class DrexResponse(BaseModel):
    """
    A Drex response read as plain data.

    nace-sdk's own response classes drop fields and answer types they don't know, so answers are kept as the dicts
    Drex sends: new answer types and fields reach the caller without a new release of this package.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any] | None = None


def require_answers(response: DrexResponse, question_ids: Sequence[str], *, probabilities: Sequence[str] = ()) -> None:
    """
    Validates required answers while retaining extra fields and future answer types.

    :param response: The parsed backend response.
    :param question_ids: Answers that must be present for this request to succeed.
    :param probabilities: Answers the caller consumes as `noul` probabilities.
    :raises nace_sdk.NaceAPIResponseValidationError: If required data is missing or invalid.
    """
    for question_id in question_ids:
        answer = response.answers.get(question_id)
        field = f"answers.{question_id}"
        if answer is not None:
            if answer.get("type") != "noul" and question_id not in probabilities:
                continue
            value = answer.get("noul")
            if (
                answer.get("type") == "noul"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value <= 1
                and math.isfinite(value)
            ):
                continue
            field += ".noul"
        raise nace_sdk.NaceAPIResponseValidationError(200, response.model_dump(), httpx.Headers(), field)


def _as_dict(question: Any) -> Any:
    # nace-sdk question objects, and those of compatible SDKs such as typesafe_sdk, are pydantic models
    dump = getattr(question, "model_dump", None)
    return dump() if callable(dump) and not isinstance(question, Mapping) else question


def validate_questions(questions: Mapping[str, Question]) -> None:
    """
    Checks the structure of the questions. Their content is checked by Drex.

    Question types other than `choice`, `score` and `noul` are accepted, so new Drex question types work without a
    new release of this package.

    :param questions: Questions keyed by question ID.
    :raises ValueError: If `questions` is empty, a question has no `type`, or a `choice` or `score` question has no
        `criteria`.
    """
    if not questions:
        msg = "`questions` must contain at least one question."
        raise ValueError(msg)
    for question_id, question in questions.items():
        # SDK question objects are validated by the SDK when they are created
        if isinstance(question, (nace_sdk.Choice, nace_sdk.Score, nace_sdk.Noul)):
            continue
        fields = _as_dict(question)
        if not isinstance(fields, Mapping):
            msg = f"Question '{question_id}' must be a dict or a question object, got {type(fields).__name__}."
            raise ValueError(msg)
        question_type = fields.get("type")
        if not isinstance(question_type, str) or not question_type:
            msg = f"Question '{question_id}' needs a `type`, such as 'choice', 'score' or 'noul'."
            raise ValueError(msg)
        if question_type in ("choice", "score") and not fields.get("criteria"):
            msg = f"Question '{question_id}' of type '{question_type}' needs `criteria`."
            raise ValueError(msg)


def to_wire(question: Question) -> dict[str, Any]:
    """
    Returns a question as Drex reads it.

    `instructions` is always a string (`""` when omitted), because Drex rejects `null` and OpenRouter requires the
    key. `None` choice descriptions are kept. Questions of other types are sent as given.

    :param question: A question dict, or an SDK `Choice`, `Score` or `Noul`.
    :returns: The question as a JSON-ready dict.
    """
    wire = dict(_as_dict(question))
    instructions = wire.get("instructions")
    if instructions is None:
        wire["instructions"] = ""
    elif not isinstance(instructions, str):
        wire["instructions"] = json.dumps(instructions)
    if wire.get("type") == "noul" and "criteria" in wire:
        criteria = {side: text for side, text in (wire["criteria"] or {}).items() if text is not None}
        if criteria:
            wire["criteria"] = criteria
        else:
            del wire["criteria"]
    return wire


def build_call(
    state: Any,
    questions: Mapping[str, Question],
    *,
    model: str,
    images: Sequence[ImageInput] | None,
    openrouter: bool,
) -> dict[str, Any]:
    """
    Builds the keyword arguments for `system_one` on a nace-sdk client.

    :param state: The text or JSON the questions are about.
    :param questions: Questions keyed by question ID.
    :param model: The model ID, sent as given.
    :param images: Images for models that read them, in the order the prompt refers to them as "Image 1", "Image 2".
    :param openrouter: Whether the request goes to OpenRouter, which takes images inside `state`.
    :returns: The `system_one` keyword arguments, including `response_model=DrexResponse`.
    :raises ValueError: If images are sent to a text-only model, or an OpenRouter model ID has no provider prefix.
    """
    if openrouter and "/" not in model:
        msg = f"OpenRouter names models with a provider prefix: use model='nace-ai/{model}'."
        raise ValueError(msg)
    if images and is_text_only(model):
        msg = f"{model} reads text only, so it can't take images. Use an image model such as model='drex-max-v1.1'."
        raise ValueError(msg)

    wire_questions = {question_id: to_wire(question) for question_id, question in questions.items()}
    wire_images = [to_wire_image(image, openrouter=openrouter) for image in images or ()]
    if wire_images:
        body = json.dumps({"model": model, "state": state, "questions": wire_questions, "images": []})
        wire_images = fit_images(wire_images, MAX_REQUEST_BYTES - len(body) - _JSON_HEADROOM)
    if openrouter and wire_images:
        state, wire_images = _openrouter_state(state, wire_images), []
    return {
        "state": state,
        "questions": wire_questions,
        "model": model,
        "images": wire_images or None,
        "response_model": DrexResponse,
    }


def _openrouter_state(state: Any, images: list[str | dict[str, str]]) -> list[dict[str, Any]]:
    # OpenRouter rejects a top-level `images` field and takes images as `image_url` parts of a `state` array
    parts: list[dict[str, Any]] = [{"type": "text", "text": state if isinstance(state, str) else json.dumps(state)}]
    parts.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
    return parts
