"""Windows UTF-8 console guard (Bug #4 regression).

Rich's box-drawing glyphs crash on Windows' default cp1252 console
(UnicodeEncodeError) — hits every Rich-output command. `_force_utf8_streams`
reconfigures stdout/stderr to UTF-8 on Windows, no-op elsewhere.
"""

from __future__ import annotations

from slancha_local.cli import _force_utf8_streams


class _FakeStream:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kw):
        self.calls.append(kw)


def test_reconfigures_to_utf8_on_windows(monkeypatch):
    monkeypatch.setattr("slancha_local.cli.sys.platform", "win32")
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr("slancha_local.cli.sys.stdout", out)
    monkeypatch.setattr("slancha_local.cli.sys.stderr", err)
    _force_utf8_streams()
    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr("slancha_local.cli.sys.platform", "linux")
    out = _FakeStream()
    monkeypatch.setattr("slancha_local.cli.sys.stdout", out)
    _force_utf8_streams()
    assert out.calls == []  # never touches streams off Windows


def test_survives_stream_without_reconfigure(monkeypatch):
    monkeypatch.setattr("slancha_local.cli.sys.platform", "win32")
    monkeypatch.setattr("slancha_local.cli.sys.stdout", object())  # no reconfigure attr
    monkeypatch.setattr("slancha_local.cli.sys.stderr", object())
    _force_utf8_streams()  # must not raise
