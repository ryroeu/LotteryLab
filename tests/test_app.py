"""Smoke tests for the Streamlit app: every view renders with defaults, no exception.

Each view file is self-bootstrapping (it puts app/ and the repo root on sys.path),
so AppTest can run any page standalone — the same way `streamlit run` would.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    "app/main.py",
    "app/views/overview.py",
    "app/views/frequency.py",
    "app/views/prove.py",
    "app/views/wheel.py",
    "app/views/ev.py",
    "app/views/simulator.py",
    "app/views/data.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    at.run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
