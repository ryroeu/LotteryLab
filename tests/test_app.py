"""Smoke-test every page of the Streamlit app (skipped without streamlit)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(
    p for p in (ROOT / "app" / "views").glob("*.py") if p.name != "__init__.py"
)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_error(page, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("app", None)
    app = AppTest.from_file(str(page), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert app.title, "every page has a title"
