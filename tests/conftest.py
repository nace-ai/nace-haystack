# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from haystack.dataclasses import ImageContent

from haystack_integrations.common.nace.images import MAX_REQUEST_BYTES

from .helpers import phone_scan_png


@pytest.fixture(scope="session")
def phone_scan() -> ImageContent:
    """A synthetic phone-photo scan that is too big to send in one request as it is."""
    data = phone_scan_png()
    assert len(data) > MAX_REQUEST_BYTES, "the fixture must be over the request limit to exercise shrinking"
    return ImageContent(base64_image=base64.b64encode(data).decode(), mime_type="image/png")
