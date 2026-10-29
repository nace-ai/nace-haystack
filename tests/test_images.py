# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import nace_sdk
import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.common.nace import images
from haystack_integrations.common.nace.images import MAX_REQUEST_BYTES, fit_images, image_types, to_wire_image

from .helpers import RED_PNG, decode_data_url, first_pixel, image_size, solid_png

REFERENCE = {"type": "workspace_file", "workspace_id": "w", "file_id": "f"}


class TestToWireImage:
    def test_any_image_type_is_sent(self):
        gif = to_wire_image(ByteStream(data=b"GIF89a...", mime_type="image/gif"), openrouter=False)
        pdf = to_wire_image(ByteStream(data=b"%PDF-1.7", mime_type="application/pdf"), openrouter=False)
        assert gif.startswith("data:image/gif;base64,")
        assert pdf.startswith("data:application/pdf;base64,")
        assert decode_data_url(pdf) == b"%PDF-1.7"

    def test_bytestream_without_mime_is_sniffed(self):
        wire = to_wire_image(ByteStream(data=solid_png((255, 0, 0))), openrouter=False)
        assert wire.startswith("data:image/png;base64,")

    def test_image_content(self):
        assert to_wire_image(RED_PNG, openrouter=False) == f"data:image/png;base64,{RED_PNG.base64_image}"

    @pytest.mark.parametrize(
        "ref",
        [
            REFERENCE,
            nace_sdk.UploadedFile(workspace_id="w", file_id="f", path="scan.png"),
            nace_sdk.WorkspaceFileSource(workspace_id="w", file_id="f"),
        ],
    )
    def test_file_reference_passes_through(self, ref):
        assert to_wire_image(ref, openrouter=False) == REFERENCE

    def test_file_reference_on_openrouter_refused(self):
        with pytest.raises(ValueError, match="inline images only"):
            to_wire_image(REFERENCE, openrouter=True)

    @pytest.mark.parametrize("value", ["data:image/png;base64,AAAA", {"type": "image_url", "url": "x"}, 42])
    def test_unsupported_input_refused(self, value):
        with pytest.raises(ValueError, match="ImageContent"):
            to_wire_image(value, openrouter=False)


def test_image_types():
    pdf = ByteStream(data=b"%PDF-1.7", mime_type="application/pdf")
    assert image_types([RED_PNG, pdf, REFERENCE]) == ["image/png", "application/pdf", "workspace_file"]


class TestFitImages:
    def test_small_images_untouched(self):
        small = [to_wire_image(RED_PNG, openrouter=False), REFERENCE]
        assert fit_images(small, budget=MAX_REQUEST_BYTES) == small

    def test_big_image_shrunk_to_fit(self, caplog, phone_scan):
        [out] = fit_images([to_wire_image(phone_scan, openrouter=False)], budget=3_000_000)
        assert len(out) <= 3_000_000
        assert out.startswith("data:image/jpeg;base64,")
        width, height = image_size(out)
        assert abs(width / height - 3024 / 4032) < 0.01
        # Still at least what Drex Max reads: it downsizes to about 1.7 megapixels itself (measured 2026-10-28)
        assert width * height >= 1_700_000
        assert "3024x4032" in caplog.text
        assert "shrunk" in caplog.text

    def test_budget_shared_and_order_kept(self, phone_scan):
        scan = to_wire_image(phone_scan, openrouter=False)
        red = to_wire_image(RED_PNG, openrouter=False)
        out = fit_images([scan, red, REFERENCE, scan], budget=3_000_000)
        assert out[1] == red
        assert out[2] == REFERENCE
        assert out[0].startswith("data:image/jpeg;base64,")
        assert out[3].startswith("data:image/jpeg;base64,")
        assert len(out[0]) + len(out[1]) + len(out[3]) <= 3_000_000

    def test_transparent_png_flattened_onto_white(self):
        from PIL import Image  # noqa: PLC0415

        noisy = Image.effect_noise((1200, 1200), 64).convert("RGBA")
        noisy.paste((0, 0, 0, 0), (0, 0, 100, 100))  # a transparent corner whose colour channels are black
        buffer = io.BytesIO()
        noisy.save(buffer, "PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        [out] = fit_images([data_url], budget=len(data_url) // 2)
        assert out.startswith("data:image/jpeg;base64,")
        assert all(channel > 240 for channel in first_pixel(out))

    def test_unopenable_data_left_alone(self):
        pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.7\n" + b"0" * 5_000_000).decode()
        assert fit_images([pdf], budget=1_000_000) == [pdf]

    def test_without_pillow_clear_error(self, monkeypatch, phone_scan):
        def missing() -> None:
            msg = "No module named 'PIL'"
            raise ImportError(msg)

        monkeypatch.setattr(images.pillow_import, "check", missing)
        with pytest.raises(ValueError, match="pip install pillow") as error:
            fit_images([to_wire_image(phone_scan, openrouter=False)], budget=3_000_000)
        assert "size=" in str(error.value)


class TestFitImagesLimits:
    def test_references_only_left_alone_even_without_room(self):
        references = [REFERENCE, dict(REFERENCE, file_id="g")]
        assert fit_images(references, budget=-5) == references

    def test_no_room_for_images_raises_clear_error(self):
        with pytest.raises(ValueError, match="too big to send with this much text"):
            fit_images([to_wire_image(RED_PNG, openrouter=False)], budget=-1)

    def test_shrinking_stops_at_the_quality_floor(self, phone_scan):
        with pytest.raises(ValueError, match="too big to send with this much text"):
            fit_images([to_wire_image(phone_scan, openrouter=False)], budget=100_000)

    def test_exif_orientation_applied_before_shrinking(self):
        from PIL import Image  # noqa: PLC0415

        landscape = Image.effect_noise((4032, 3024), 40).convert("RGB")
        exif = Image.Exif()
        exif[0x0112] = 6  # the phone held upright: rotate 90 degrees to display
        buffer = io.BytesIO()
        landscape.save(buffer, "JPEG", quality=95, exif=exif)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        [out] = fit_images([data_url], budget=len(data_url) // 2)
        width, height = image_size(out)
        assert width < height


def test_octet_stream_bytestream_is_sniffed():
    image = ByteStream(data=solid_png((255, 0, 0)), mime_type="application/octet-stream")
    assert to_wire_image(image, openrouter=False).startswith("data:image/png;base64,")
    assert image_types([image]) == ["image/png"]


def test_16_bit_grayscale_scan_keeps_its_content_when_shrunk():
    # Scanners often save 16-bit grayscale PNGs; converting those straight to RGB clips every pixel to white
    from PIL import Image  # noqa: PLC0415  # test-only dependency

    # Grey paper with grain (values around 30000 of 65535) and a dark band of text, above the 1-megapixel floor
    grain = Image.effect_noise((1600, 1200), 20).point(lambda value: 110 + value // 8)
    page = grain.convert("I").point(lambda value: value * 256)
    page.paste(Image.new("I", (1200, 160), 1000), (200, 520))
    page = page.convert("I;16")
    buffer = io.BytesIO()
    page.save(buffer, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    [shrunk] = fit_images([data_url], budget=len(data_url) // 2)

    with Image.open(io.BytesIO(decode_data_url(shrunk))) as sent:
        gray = sent.convert("L")
        band = gray.getpixel((gray.width // 2, gray.height * 600 // 1200))
        paper = gray.getpixel((gray.width // 10, gray.height // 10))
    assert band < 30
    assert 100 < paper < 135
