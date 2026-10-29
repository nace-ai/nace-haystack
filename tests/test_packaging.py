# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from hatchling.builders.sdist import SdistBuilder

ROOT = Path(__file__).parent.parent

APACHE_HEADER = [
    "# SPDX-FileCopyrightText: 2026-present Nace AI",
    "#",
    "# SPDX-License-Identifier: Apache-2.0",
]


def test_every_source_file_has_the_apache_header():
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py")]:
        head = path.read_text(encoding="utf-8").splitlines()[:3]
        if not head:  # an empty __init__.py
            continue
        assert head == APACHE_HEADER, path


def test_pydantic_is_a_declared_dependency():
    assert '"pydantic>=2.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_sdist_ships_only_the_package_and_its_tests():
    # The source distribution is uploaded to PyPI next to the wheel, so internal notes must never be in it
    shipped = {
        included.distribution_path.replace("\\", "/")  # Windows paths
        for included in SdistBuilder(str(ROOT)).recurse_included_files()
    }
    top_level = {path.split("/", 1)[0] for path in shipped}
    assert top_level <= {"src", "tests", "README.md", "LICENSE.txt", "CHANGELOG.md", "pyproject.toml", ".gitignore"}
    assert "src/haystack_integrations/components/routers/nace/text_router.py" in shipped


def test_license_names_nace_as_the_copyright_holder():
    text = (ROOT / "LICENSE.txt").read_text(encoding="utf-8")
    assert "Copyright 2026-present Nace AI" in text
    assert "deepset" not in text
