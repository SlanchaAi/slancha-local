"""TUI smoke test: app constructs cleanly + percentile helper works."""

from __future__ import annotations

import pytest

from slancha_local.tui.app import SlanchaTUI, _percentile, _short_time


def test_tui_constructs(tmp_path):
    app = SlanchaTUI(proxy_url="http://127.0.0.1:9999", traces_root=tmp_path)
    assert app is not None
    assert app._proxy_url == "http://127.0.0.1:9999"


def test_percentile_empty():
    assert _percentile([], 95) == 0


@pytest.mark.parametrize(
    "values,p,expected",
    [
        ([100], 95, 100),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50, 6),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 10),
    ],
)
def test_percentile_ranks(values, p, expected):
    assert _percentile(values, p) == expected


def test_short_time_iso():
    assert len(_short_time("2026-05-09T14:23:11.482Z")) == 8  # HH:MM:SS


def test_short_time_garbage_does_not_crash():
    out = _short_time("garbage")
    assert isinstance(out, str)
