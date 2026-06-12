"""Smoke tests for the Streamlit app: every view renders with defaults."""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

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
    """Render one Streamlit page with default widget values."""
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    at.run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
