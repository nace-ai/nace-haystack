# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic images for the tests: no real documents, no network."""

import base64
import io
import struct
import zlib

from haystack.dataclasses import ImageContent


def solid_png(rgb: tuple[int, int, int], size: int = 8) -> bytes:
    """A tiny solid-colour PNG, built by hand so the fixtures don't need Pillow."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b"")
    )


RED_PNG = ImageContent(base64_image=base64.b64encode(solid_png((255, 0, 0))).decode(), mime_type="image/png")
BLUE_PNG = ImageContent(base64_image=base64.b64encode(solid_png((0, 0, 255))).decode(), mime_type="image/png")


def phone_scan_png(width: int = 3024, height: int = 4032) -> bytes:
    """A synthetic invoice photographed at phone resolution: paper grain on every pixel makes it a multi-MB PNG."""
    from PIL import Image, ImageDraw  # noqa: PLC0415  # test-only dependency

    grain = Image.effect_noise((width, height), 24).point(lambda value: 228 + value // 12)
    page = Image.merge("RGB", (grain, grain, grain.point(lambda value: max(value - 6, 0))))
    draw = ImageDraw.Draw(page)
    scale = width / 1000
    draw.text((60 * scale, 50 * scale), "INVOICE INV-1042", fill=(20, 20, 20), font_size=int(40 * scale))
    for line in range(40):
        draw.text(
            (60 * scale, (130 + line * 20) * scale),
            f"Line {line + 1:02d}  Consulting services  2 x $142.00",
            fill=(40, 40, 40),
            font_size=int(14 * scale),
        )
    draw.text((60 * scale, 960 * scale), "TOTAL DUE: $4,212.00", fill=(150, 0, 0), font_size=int(22 * scale))
    buffer = io.BytesIO()
    page.save(buffer, "PNG")
    return buffer.getvalue()


def decode_data_url(data_url: str) -> bytes:
    """The raw bytes inside a `data:` URL."""
    return base64.b64decode(data_url.partition(",")[2])


def image_size(data_url: str) -> tuple[int, int]:
    """Pixel size of the image inside a `data:` URL."""
    from PIL import Image  # noqa: PLC0415  # test-only dependency

    with Image.open(io.BytesIO(decode_data_url(data_url))) as image:
        return image.size


def first_pixel(data_url: str) -> tuple[int, ...]:
    """Colour of the top-left pixel of the image inside a `data:` URL."""
    from PIL import Image  # noqa: PLC0415  # test-only dependency

    with Image.open(io.BytesIO(decode_data_url(data_url))) as image:
        return image.convert("RGB").getpixel((0, 0))
