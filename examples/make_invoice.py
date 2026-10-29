# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Make a synthetic, phone-photo-sized scan of a paper invoice.

Everything on the page is fictional: the vendor, the customer, the line items and
the invoice number (INV-1042) are made up. The image is deliberately large
(3024 x 4032 pixels, about 12 megapixels) so the example can show what happens
when an attachment is too big to send inline.

Regenerate the example file with:

    python examples/make_invoice.py
"""

import random
from pathlib import Path

import PIL.Image
from PIL import ImageDraw, ImageFont

PAPER = (246, 244, 238)
INK = (30, 30, 30)
RED = (190, 30, 30)
RULE = (150, 150, 150)

LINE_COUNT = 40
TOTAL_CENTS = 421_200  # TOTAL DUE: $4,212.00

# Grain: width * height // GRAIN_DIVISOR pixels get a random grey value from GRAIN_RANGE.
# A smaller divisor adds more specks, which makes the PNG larger. At 40 the PNG is about
# 2.2 MB, under the 4 MiB request limit, so 10 is used: the PNG is about 5.5 MB.
GRAIN_DIVISOR = 10
GRAIN_RANGE = (200, 240)


def _line_items() -> list[tuple[int, int]]:
    """
    Return (quantity, unit price in cents) for each invoice line.

    The last line balances the invoice so that the lines add up to TOTAL_CENTS.
    """
    rng = random.Random(1)
    items = []
    for _ in range(LINE_COUNT - 1):
        quantity = rng.randint(1, 4)
        unit_cents = rng.randint(2_000, 5_000)  # $20.00 to $50.00
        items.append((quantity, unit_cents))
    remainder = TOTAL_CENTS - sum(quantity * unit for quantity, unit in items)
    if remainder <= 0:
        msg = "line items exceed TOTAL_CENTS; lower the unit price range"
        raise ValueError(msg)
    items.append((1, remainder))
    return items


def _add_grain(page: PIL.Image.Image, rng: random.Random) -> None:
    """Scatter light grey specks over the page, in place."""
    width, height = page.size
    low, high = GRAIN_RANGE
    pixels = page.load()
    for _ in range(width * height // GRAIN_DIVISOR):
        x = rng.randrange(width)
        y = rng.randrange(height)
        grey = rng.randint(low, high)
        pixels[x, y] = (grey, grey, grey)


def _draw_stamp(
    page: PIL.Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    right: int,
    top: int,
) -> None:
    """
    Paste a red, boxed note onto the page, tilted a few degrees.

    The right edge of the note lands on x = ``right`` and its top edge on y = ``top``.
    """
    pad, border = 40, 6
    left, text_top, text_right, text_bottom = font.getbbox(text)
    box_w = text_right - left + 2 * pad
    box_h = text_bottom - text_top + 2 * pad

    # Draw on a transparent layer so the whole note can be rotated.
    layer = PIL.Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    red = (*RED, 255)
    draw.rectangle((0, 0, box_w - 1, box_h - 1), outline=red, width=border)
    draw.text((pad - left, pad - text_top), text, font=font, fill=red)
    layer = layer.rotate(-3, resample=PIL.Image.Resampling.BICUBIC, expand=True)

    # The layer's alpha channel is the paste mask, so the paper shows around the note.
    page.paste(layer, (right - layer.width, top), layer)


def make_invoice(width: int = 3024, height: int = 4032) -> "PIL.Image.Image":
    """
    Draw a synthetic scanned invoice and return it as an RGB image.

    Font sizes scale with ``width``. The vertical layout is tuned for the default
    3024 x 4032 size. ``ImageFont.load_default(size=...)`` needs Pillow 10.1 or later.
    """
    title_size = width // 16
    header_size = width // 30
    body_size = width // 66
    total_size = width // 24
    stamp_size = width // 40
    title_font = ImageFont.load_default(size=title_size)
    header_font = ImageFont.load_default(size=header_size)
    body_font = ImageFont.load_default(size=body_size)
    total_font = ImageFont.load_default(size=total_size)
    stamp_font = ImageFont.load_default(size=stamp_size)

    page = PIL.Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(page)
    margin = width // 12

    # Header block.
    y = height // 14
    draw.text((margin, y), "INVOICE INV-1042", font=title_font, fill=INK)
    y += int(title_size * 1.4)
    draw.text((margin, y), "Example Consulting Ltd", font=header_font, fill=INK)
    y += int(header_size * 1.4)
    draw.text((margin, y), "Bill to: Alex Example", font=body_font, fill=INK)
    y += int(body_size * 2.5)
    draw.line((margin, y, width - margin, y), fill=RULE, width=4)
    y += int(body_size * 1.2)

    # One row per line item.
    row_height = int(body_size * 1.4)
    for number, (quantity, unit_cents) in enumerate(_line_items(), start=1):
        price = f"${unit_cents / 100:,.2f}"
        text = f"Line {number:02d}  Consulting services  {quantity} x {price}"
        draw.text((margin, y), text, font=body_font, fill=INK)
        y += row_height

    # Total in bold red. Pillow's default font has no bold weight, so the glyphs get a stroke.
    y += int(body_size * 1.2)
    draw.line((margin, y, width - margin, y), fill=RULE, width=4)
    y += int(body_size * 1.5)
    total_text = f"TOTAL DUE: ${TOTAL_CENTS / 100:,.2f}"
    draw.text(
        (margin, y),
        total_text,
        font=total_font,
        fill=RED,
        stroke_width=3,
        stroke_fill=RED,
    )

    # Stamp-style note under the total, aligned to the right margin.
    stamp_text = "PAID TWICE - Mar 3 and Mar 5"  # the default font has no em dash glyph
    _draw_stamp(page, stamp_text, stamp_font, right=width - margin, top=y + int(total_size * 1.6))

    _add_grain(page, random.Random(2))
    return page


if __name__ == "__main__":
    output = Path(__file__).parent / "data" / "scanned_invoice.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    invoice = make_invoice()
    invoice.save(output, format="PNG")
    size_bytes = output.stat().st_size
    print(f"Wrote {output}")
    print(f"Pixel size: {invoice.width} x {invoice.height}")
    print(f"File size: {size_bytes / 1_000_000:.2f} MB ({size_bytes:,} bytes)")
