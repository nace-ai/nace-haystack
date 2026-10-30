# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import nace_sdk
import pytest

from haystack_integrations.common.nace.errors import BILLING_URL, OPENROUTER_CREDITS_URL, readable_error


def sdk_error(status: int, body: object, headers: dict[str, str] | None = None) -> nace_sdk.NaceError:
    """The error nace-sdk raises for this response, so the tests exercise its real error parsing."""

    def handler(_: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body, headers=headers or {})
        return httpx.Response(status, json=body, headers=headers or {})

    client = nace_sdk.NaceClient(
        api_key="nace_sk_test", transport=httpx.MockTransport(handler), retry=nace_sdk.RetryPolicy(max_retries=0)
    )
    try:
        client.system_one(state="hi", questions={"q": {"type": "noul", "instructions": "x"}})
    except nace_sdk.NaceError as error:
        return error
    pytest.fail("the mocked response didn't raise")


def invalid(*issues: tuple[str, str], message: str = "the request is invalid") -> dict:
    return {
        "error": {
            "type": "invalid_request_error",
            "message": message,
            "issues": [{"path": path, "message": text} for path, text in issues],
        }
    }


def test_422_lists_each_issue():
    error = readable_error(
        sdk_error(
            422,
            invalid(("questions.q.criteria", "must have at least 2 labels"), ("model", 'unknown model "jev-latest"')),
        ),
        "Drex",
    )
    text = str(error)
    assert text.startswith("422 from Drex: the request is invalid")
    assert "  questions.q.criteria: must have at least 2 labels" in text
    assert '  model: unknown model "jev-latest"' in text
    assert isinstance(error, nace_sdk.NaceUnprocessableEntityError)
    assert error.issues[1]["path"] == "model"


def test_single_issue_not_repeated():
    body = invalid(("model", 'unknown model "jev-latest"'), message='model: unknown model "jev-latest"')
    text = str(readable_error(sdk_error(422, body), "Drex"))
    assert text.count('unknown model "jev-latest"') == 1


def test_402_names_billing():
    error = readable_error(sdk_error(402, {"error": {"message": "no available funds"}}), "Drex")
    assert isinstance(error, nace_sdk.NaceInsufficientCreditError)
    assert f"Add funds at {BILLING_URL}" in str(error)
    assert BILLING_URL == "https://console.nace.ai/dashboard/billing"


def test_openrouter_prefix_and_unwrap():
    error = readable_error(sdk_error(422, {"error": {"message": 'HTTP 422: {"detail": "bad"}'}}), "OpenRouter")
    assert str(error) == "422 from OpenRouter: bad"


def test_413_is_readable():
    text = str(readable_error(sdk_error(413, "<html>Request Entity Too Large</html>"), "Drex"))
    assert text.startswith("413 from Drex:")
    assert "too large" in text
    assert "<html>" not in text
    assert "MiB" not in text
    assert "MB" not in text


def test_pdf_refusal_points_to_converter():
    body = invalid(("images.1", "must be a PNG, JPEG or WebP base64 image"))
    text = str(readable_error(sdk_error(422, body), "Drex", image_types=["image/png", "application/pdf"]))
    assert "Image 2 is a PDF" in text
    assert "PDFToImageContent" in text


def test_no_hint_for_other_issues():
    body = invalid(("images.0", "must be a PNG, JPEG or WebP base64 image"))
    text = str(readable_error(sdk_error(422, body), "Drex", image_types=["image/png"]))
    assert "PDFToImageContent" not in text


def test_request_id_kept():
    error = readable_error(sdk_error(429, {"error": {"message": "slow down"}}, {"x-request-id": "req_123"}), "Drex")
    assert error.request_id == "req_123"
    assert str(error).endswith("(request_id=req_123)")


def test_catchable_as_the_sdk_class():
    with pytest.raises(nace_sdk.NaceRateLimitError):
        raise readable_error(sdk_error(429, {"error": {"message": "slow down"}}), "Drex")


def test_other_status_keeps_base_class():
    error = readable_error(sdk_error(418, {"error": {"message": "teapot"}}), "Drex")
    assert type(error) is nace_sdk.NaceAPIError
    assert str(error) == "418 from Drex: teapot"


def test_non_api_errors_unchanged():
    timeout = nace_sdk.NaceAPITimeoutError(5.0)
    assert readable_error(timeout, "Drex") is timeout


def test_413_with_a_pdf_points_to_converter():
    text = str(readable_error(sdk_error(413, "<html>Too Large</html>"), "Drex", image_types=["application/pdf"]))
    assert "PDFToImageContent" in text


def test_413_without_images_says_to_shorten_the_text():
    text = str(readable_error(sdk_error(413, "<html>Too Large</html>"), "Drex"))
    assert "image" not in text.lower()
    assert "shorten the text" in text.lower()


def test_402_on_openrouter_points_to_openrouter_credits():
    error = readable_error(sdk_error(402, {"error": {"message": "Insufficient credits"}}), "OpenRouter")
    assert OPENROUTER_CREDITS_URL in str(error)
    assert BILLING_URL not in str(error)


def test_html_error_page_is_summarised():
    page = "<!DOCTYPE html><html><body>" + "x" * 5000 + "</body></html>"
    error = readable_error(sdk_error(502, page), "Drex")
    assert "<html" not in str(error).lower()
    assert "Drex returned an HTML error page (HTTP 502)" in str(error)


def test_long_plain_message_is_truncated():
    error = readable_error(sdk_error(500, "y" * 5000), "Drex")
    assert len(error.message) <= 501
    assert error.message.endswith("…")
    assert "y" * 501 not in str(error)
    assert "…" in str(error)


def test_404_names_the_url_so_a_wrong_base_url_shows():
    original = sdk_error(404, "Not Found")
    error = readable_error(original, "Drex")
    assert str(error).startswith("404 from Drex: Not Found")
    assert "https://console.nace.ai/v1/systemone" in str(error)
    assert "base URL" in str(error)
    assert error.idempotency_key == original.idempotency_key


def test_other_errors_keep_their_short_form():
    assert str(readable_error(sdk_error(418, "teapot"), "Drex")) == "418 from Drex: teapot"


@pytest.mark.parametrize(
    "issues",
    [
        [{"path": "questions.q", "message": "x" * 100_000}],
        [{"path": f"questions.q{i}", "message": "invalid"} for i in range(1000)],
    ],
)
def test_structured_issues_have_a_total_display_budget(issues):
    original = sdk_error(422, {"error": {"message": "invalid", "issues": issues}})
    rewritten = readable_error(original, "Drex")
    assert len(rewritten.message) <= 501
    assert rewritten.message.endswith("…")
    assert rewritten.body == original.body
    assert rewritten.issues == original.issues
