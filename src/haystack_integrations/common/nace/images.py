# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import filetype
import nace_sdk
from haystack import logging
from haystack.dataclasses import ByteStream, ImageContent
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

ImageInput: TypeAlias = (
    ImageContent | ByteStream | nace_sdk.UploadedFile | nace_sdk.WorkspaceFileSource | Mapping[str, str]
)

# Drex's documented request size limit. It is used only to decide when inline images must be shrunk; Drex enforces
# the real limit and explains it when a request is too large.
MAX_REQUEST_BYTES = 4 * 1024 * 1024

_JPEG_QUALITY = 90
_MAX_SHRINK_ROUNDS = 6
# Below about 1 megapixel, small print on a scanned page stops being readable
_MIN_SHRUNK_PIXELS = 1_000_000


def _sniff(data: bytes) -> str:
    guess = filetype.guess(data)
    return guess.mime if guess else "application/octet-stream"


def _declared(mime_type: str | None) -> str | None:
    # `application/octet-stream` says only "some bytes", which is what loaders such as a plain file read produce
    return None if mime_type == "application/octet-stream" else mime_type


def _reference(image: Any) -> dict[str, str] | None:
    if isinstance(image, (nace_sdk.UploadedFile, nace_sdk.WorkspaceFileSource)):
        return {"type": "workspace_file", "workspace_id": str(image.workspace_id), "file_id": str(image.file_id)}
    if isinstance(image, Mapping) and image.get("type") == "workspace_file":
        return {"type": "workspace_file", "workspace_id": str(image["workspace_id"]), "file_id": str(image["file_id"])}
    return None


def to_wire_image(image: ImageInput, *, openrouter: bool) -> str | dict[str, str]:
    """
    Converts an image to the form Drex reads.

    Images are sent with their own MIME type; Drex decides which types it reads and explains a refusal.

    :param image: An `ImageContent`, a `ByteStream`, or a reference to a file uploaded to Drex with nace-sdk.
    :param openrouter: Whether the request goes to OpenRouter, which takes inline images only.
    :returns: A `data:` URL, or a `workspace_file` reference.
    :raises ValueError: If the image is a file reference on OpenRouter, or isn't a supported input.
    """
    if isinstance(image, ImageContent):
        mime_type = _declared(image.mime_type) or _sniff(base64.b64decode(image.base64_image))
        return f"data:{mime_type};base64,{image.base64_image}"
    if isinstance(image, ByteStream):
        mime_type = _declared(image.mime_type) or _sniff(image.data)
        return f"data:{mime_type};base64,{base64.b64encode(image.data).decode()}"
    reference = _reference(image)
    if reference is None:
        msg = (
            f"Unsupported image {type(image).__name__}. Pass an ImageContent, a ByteStream, or a file uploaded to "
            "Drex with nace-sdk (an UploadedFile, a WorkspaceFileSource, or a "
            '{"type": "workspace_file", "workspace_id": ..., "file_id": ...} dict).'
        )
        raise ValueError(msg)
    if openrouter:
        msg = "OpenRouter accepts inline images only, not files uploaded to Drex. Pass the image itself instead."
        raise ValueError(msg)
    return reference


def image_types(images: Sequence[ImageInput]) -> list[str]:
    """
    Lists the MIME type of each image, so error messages can name the one Drex refused.

    :param images: The images as the caller passed them.
    :returns: One MIME type per image, or `"workspace_file"` for a reference to an uploaded file.
    """
    types = []
    for image in images:
        if isinstance(image, ImageContent):
            types.append(_declared(image.mime_type) or _sniff(base64.b64decode(image.base64_image)))
        elif isinstance(image, ByteStream):
            types.append(_declared(image.mime_type) or _sniff(image.data))
        else:
            types.append("workspace_file")
    return types


def fit_images(images: list[str | dict[str, str]], budget: int) -> list[str | dict[str, str]]:
    """
    Shrinks inline images that don't fit in one request, keeping everything else as it is.

    Drex Max downsizes every image to about 1.6 megapixels itself, so a shrunk image that still fits keeps the
    detail the model reads. Images that fit are sent untouched, so a model that reads more detail isn't held back.
    An image is never shrunk below about 1 megapixel, where small print stops being readable: when the text leaves
    too little room, a clear error says so instead.

    :param images: `data:` URLs and `workspace_file` references, in request order.
    :param budget: How many characters the inline images may take in the request in total.
    :returns: The images in the same order. Data URLs over their share of `budget` are re-encoded as JPEG and scaled
        down until they fit. References, and data Pillow can't open (such as a PDF), are returned unchanged.
    :raises ValueError: If an image must be shrunk and Pillow isn't installed, or can't fit without going below
        the quality floor.
    """
    inline = [index for index, image in enumerate(images) if isinstance(image, str)]
    if not inline or sum(len(images[index]) for index in inline) <= budget:
        return list(images)
    share = budget // len(inline)
    oversized = [index for index in inline if len(images[index]) > share]
    left = budget - sum(len(images[index]) for index in inline if index not in oversized)
    share = left // len(oversized)
    fitted = list(images)
    for index in oversized:
        fitted[index] = _shrink(str(images[index]), share, number=index + 1)
    return fitted


def _no_room(number: int) -> ValueError:
    msg = (
        f"Image {number} is too big to send with this much text: it would have to shrink below about 1 megapixel to "
        "fit Drex's request limit. Shorten the text, send fewer images, or pass big images as files uploaded to Drex "
        "with nace-sdk, which don't count toward the request size."
    )
    return ValueError(msg)


def _shrink(data_url: str, share: int, *, number: int) -> str:
    try:
        pillow_import.check()
    except ImportError as error:
        msg = (
            "An image is too big to send to Drex in one request, and shrinking it needs Pillow. "
            "Run 'pip install pillow', or downscale images before this component, for example with "
            "ImageFileToImageContent(size=(2048, 2048)) or PDFToImageContent(size=(2048, 2048))."
        )
        raise ValueError(msg) from error

    raw = base64.b64decode(data_url.partition(",")[2])
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            # Phone photos store their rotation in EXIF, which JPEG re-encoding drops: apply it to the pixels first
            original = _flatten(ImageOps.exif_transpose(opened))
    except OSError:
        # Not an image Pillow can read, such as a PDF. Drex refuses it with a readable error.
        return data_url
    if share <= 0:
        raise _no_room(number)

    scale = 1.0
    image = original
    shrunk = _jpeg(image)
    for _ in range(_MAX_SHRINK_ROUNDS):
        if len(shrunk) <= share:
            break
        scale *= math.sqrt(share / len(shrunk)) * 0.95
        size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
        if size[0] * size[1] < _MIN_SHRUNK_PIXELS:
            raise _no_room(number)
        image = original.resize(size, Image.Resampling.LANCZOS)
        shrunk = _jpeg(image)
    if len(shrunk) > share:
        raise _no_room(number)
    logger.warning(
        "Image {original} ({before:.1f} MB) -> {size} ({after:.1f} MB), shrunk to fit Drex's request limit.",
        original=f"{original.width}x{original.height}",
        before=len(raw) / 1e6,
        size=f"{image.width}x{image.height}",
        after=len(shrunk) * 3 / 4 / 1e6,
    )
    return shrunk


def _flatten(image: "Image.Image") -> "Image.Image":
    # JPEG has no transparency: paint transparent areas white rather than letting them turn black.
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        page = Image.new("RGB", rgba.size, (255, 255, 255))
        page.paste(rgba, mask=rgba.getchannel("A"))
        return page
    # 16-bit and 32-bit grayscale (common in scanner output) would be clipped at 255, turning the page white.
    # Scale it down to 8 bits instead.
    if image.mode.startswith("I;16") or image.mode == "I":
        image = image.convert("I").point(lambda value: value / 256).convert("L")
    return image.convert("RGB")


def _jpeg(image: "Image.Image") -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
