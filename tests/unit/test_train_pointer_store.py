"""Unit tests for ``slancha_local.train.pointer_store``.

These exercise the durability story (atomic ACTIVE rewrite, prune-on-promote,
rollback, version validation) and the inert-by-default fallback that lets
loaders coexist with their legacy fixed-path artifact when no ACTIVE pointer
exists yet.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from slancha_local.train.pointer_store import (
    ACTIVE_FILENAME,
    PointerStore,
    PointerStoreError,
    _atomic_write_bytes,
    _atomic_write_text,
    new_version,
)

# -------- helpers --------


def _store(tmp_path: Path, **kwargs) -> PointerStore:
    return PointerStore(tmp_path / "store", **kwargs)


# -------- new_version --------


def test_new_version_format_is_utc_second_precision():
    v = new_version()
    assert len(v) == 16  # YYYYMMDDTHHMMSSZ
    assert v.endswith("Z")
    assert v[8] == "T"


def test_new_version_monotonic_when_called_in_order():
    from datetime import UTC, datetime

    v1 = new_version(datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC))
    v2 = new_version(datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC))
    assert v1 < v2


# -------- inert default (no ACTIVE pointer) --------


def test_active_version_is_none_when_no_pointer(tmp_path):
    store = _store(tmp_path)
    assert store.active_version("classifier-head") is None


def test_active_path_is_none_when_no_pointer(tmp_path):
    """Inert default — loaders fall back to legacy fixed path."""
    store = _store(tmp_path)
    assert store.active_path("classifier-head", "mmbert_tl_cluster.bin") is None


def test_versions_returns_empty_list_for_unknown_component(tmp_path):
    store = _store(tmp_path)
    assert store.versions("classifier-head") == []


# -------- write_candidate + promote happy path --------


def test_write_candidate_creates_version_dir_and_files(tmp_path):
    store = _store(tmp_path)
    vdir = store.write_candidate(
        "classifier-head",
        "20260101T000000Z",
        {"mmbert_tl_cluster.bin": b"\x00\x01\x02", "mapping.json": b'{"0":"math"}'},
    )
    assert vdir.is_dir()
    assert (vdir / "mmbert_tl_cluster.bin").read_bytes() == b"\x00\x01\x02"
    assert (vdir / "mapping.json").read_bytes() == b'{"0":"math"}'


def test_promote_flips_active_pointer(tmp_path):
    store = _store(tmp_path)
    store.write_candidate("classifier-head", "20260101T000000Z", {"a.bin": b"v1"})
    store.promote("classifier-head", "20260101T000000Z")
    assert store.active_version("classifier-head") == "20260101T000000Z"
    p = store.active_path("classifier-head", "a.bin")
    assert p is not None
    assert p.read_bytes() == b"v1"


def test_active_pointer_is_a_regular_file_not_a_symlink(tmp_path):
    """Per design: file (not symlink) for portability + uniform durability."""
    store = _store(tmp_path)
    store.write_candidate("classifier-head", "20260101T000000Z", {"a.bin": b"v"})
    store.promote("classifier-head", "20260101T000000Z")
    pointer = store.component_dir("classifier-head") / ACTIVE_FILENAME
    assert pointer.is_file()
    assert not pointer.is_symlink()


def test_promote_then_promote_replaces_pointer(tmp_path):
    store = _store(tmp_path)
    for v, payload in [("20260101T000000Z", b"v1"), ("20260101T000001Z", b"v2")]:
        store.write_candidate("classifier-head", v, {"a.bin": payload})
        store.promote("classifier-head", v)
    assert store.active_version("classifier-head") == "20260101T000001Z"
    p = store.active_path("classifier-head", "a.bin")
    assert p is not None
    assert p.read_bytes() == b"v2"


# -------- promote refuses unwritten versions --------


def test_promote_unwritten_version_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(PointerStoreError, match="does not exist"):
        store.promote("classifier-head", "20260101T000000Z")


def test_promote_invalid_version_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(PointerStoreError, match="invalid version"):
        store.promote("classifier-head", "../etc/passwd")
    with pytest.raises(PointerStoreError, match="invalid version"):
        store.promote("classifier-head", "")
    with pytest.raises(PointerStoreError, match="invalid version"):
        store.promote("classifier-head", "/abs/path")


def test_write_candidate_rejects_path_traversal_filenames(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(PointerStoreError, match="invalid candidate filename"):
        store.write_candidate("c", "v1", {"../etc/passwd": b"x"})
    with pytest.raises(PointerStoreError, match="invalid candidate filename"):
        store.write_candidate("c", "v1", {"sub/dir/file": b"x"})
    with pytest.raises(PointerStoreError, match="invalid candidate filename"):
        store.write_candidate("c", "v1", {ACTIVE_FILENAME: b"x"})


# -------- crash-simulation: ACTIVE rewrite is atomic --------


def test_active_rewrite_is_atomic_under_simulated_mid_rename_crash(tmp_path):
    """Verify-load-before-commit relies on this — if the rename is non-atomic,
    a crash mid-promote could leave us with a corrupt ACTIVE file pointing
    nowhere, breaking the loader for everybody.
    """
    store = _store(tmp_path)
    store.write_candidate("c", "20260101T000000Z", {"a.bin": b"v1"})
    store.promote("c", "20260101T000000Z")
    store.write_candidate("c", "20260101T000001Z", {"a.bin": b"v2"})

    real_replace = os.replace

    def replace_then_crash(src, dst):
        # Simulate a crash AFTER tmpfile is written but BEFORE rename lands.
        # We assert the tmpfile is on disk (write succeeded) then raise
        # without performing the rename — the pre-existing ACTIVE must be
        # untouched.
        assert Path(src).exists(), "tmpfile should exist before rename"
        raise RuntimeError("simulated crash mid-rename")

    with patch("slancha_local.train.pointer_store.os.replace", side_effect=replace_then_crash):
        with pytest.raises(RuntimeError, match="simulated crash"):
            store.promote("c", "20260101T000001Z")

    # The active pointer must still point at the old version, NOT a partial
    # or empty file.
    assert store.active_version("c") == "20260101T000000Z"
    assert real_replace is os.replace  # sanity — patch lifted


def test_crash_during_promote_does_not_leave_dangling_tmpfile(tmp_path):
    """Cosmetic durability: a crashed promote MAY leave a .tmp on disk
    (POSIX rename guarantees nothing about the source on failure), but
    that tmpfile MUST be ignored by readers — it must NOT confuse
    active_version() into reading garbage.
    """
    store = _store(tmp_path)
    store.write_candidate("c", "20260101T000000Z", {"a.bin": b"v1"})
    store.promote("c", "20260101T000000Z")

    pointer = store.component_dir("c") / ACTIVE_FILENAME
    # Plant a stale .tmp manually as if a prior crash left it
    (pointer.with_suffix(pointer.suffix + ".tmp")).write_text("garbage-version\n")

    assert store.active_version("c") == "20260101T000000Z"


# -------- K-version retention --------


def test_promote_prunes_to_keep_versions(tmp_path):
    store = _store(tmp_path, keep_versions=2)
    versions = [f"20260101T00000{i}Z" for i in range(5)]
    for v in versions:
        store.write_candidate("c", v, {"a.bin": v.encode()})
        store.promote("c", v)

    on_disk = store.versions("c")
    # keep the 2 newest (= keep_active + 1 prior); older 3 are gone.
    assert on_disk == versions[-2:]


def test_promote_never_prunes_the_active_version_even_if_it_falls_outside_k(tmp_path):
    """Defensive: if a caller promotes an older version explicitly (e.g.
    rollback workflow that re-promotes a vintage candidate), the prune
    pass must not yank the floor out from under it.
    """
    store = _store(tmp_path, keep_versions=2)
    older = "20260101T000000Z"
    newer = [f"20260101T00000{i}Z" for i in range(1, 4)]
    for v in [older, *newer]:
        store.write_candidate("c", v, {"a.bin": b"x"})
        store.promote("c", v)
    # Now re-promote the older version (rollback-via-promote scenario)
    # — first re-write it because the prune cycle dropped it.
    store.write_candidate("c", older, {"a.bin": b"x-revived"})
    store.promote("c", older)

    on_disk = store.versions("c")
    assert older in on_disk, "active version must never be pruned"


def test_keep_versions_must_be_positive(tmp_path):
    with pytest.raises(PointerStoreError, match="keep_versions"):
        PointerStore(tmp_path, keep_versions=0)


# -------- rollback --------


def test_rollback_points_active_to_prior_version(tmp_path):
    store = _store(tmp_path, keep_versions=5)
    for v in ["20260101T000000Z", "20260101T000001Z", "20260101T000002Z"]:
        store.write_candidate("c", v, {"a.bin": v.encode()})
        store.promote("c", v)
    rolled_to = store.rollback("c")
    assert rolled_to == "20260101T000001Z"
    assert store.active_version("c") == "20260101T000001Z"


def test_rollback_with_no_prior_version_removes_active_pointer(tmp_path):
    """No prior version on disk ⇒ ACTIVE removed ⇒ loader falls back to
    legacy fixed path (back-compat fully restored).
    """
    store = _store(tmp_path)
    store.write_candidate("c", "20260101T000000Z", {"a.bin": b"v"})
    store.promote("c", "20260101T000000Z")
    assert store.rollback("c") is None
    assert store.active_version("c") is None


def test_rollback_with_no_active_is_noop(tmp_path):
    """Rolling back when there's nothing active and nothing on disk should
    not crash — operator may invoke rollback defensively.
    """
    store = _store(tmp_path)
    assert store.rollback("c") is None


# -------- active_path corruption handling --------


def test_active_path_returns_none_when_version_dir_missing(tmp_path):
    """If ACTIVE points at a version dir that was deleted out-of-band
    (operator yanked it, fsck removed it, etc.), the loader must fall
    back gracefully instead of returning a path that breaks on read.
    """
    store = _store(tmp_path)
    store.write_candidate("c", "20260101T000000Z", {"a.bin": b"v"})
    store.promote("c", "20260101T000000Z")
    # Yank the version dir out from under the pointer
    vdir = store.version_dir("c", "20260101T000000Z")
    (vdir / "a.bin").unlink()
    vdir.rmdir()

    assert store.active_path("c", "a.bin") is None


def test_active_version_ignores_invalid_pointer_contents(tmp_path):
    """If ACTIVE was corrupted to hold a path-traversal or garbage string,
    we must return None instead of validating it and handing it to the
    loader.
    """
    store = _store(tmp_path)
    store.component_dir("c").mkdir(parents=True)
    (store.component_dir("c") / ACTIVE_FILENAME).write_text("../etc/passwd\n")
    assert store.active_version("c") is None


def test_active_version_returns_none_for_empty_pointer(tmp_path):
    store = _store(tmp_path)
    store.component_dir("c").mkdir(parents=True)
    (store.component_dir("c") / ACTIVE_FILENAME).write_text("")
    assert store.active_version("c") is None


# -------- components are isolated --------


def test_components_have_independent_active_pointers(tmp_path):
    """P3 future-proofing: specialists ship as their own components in the
    same store; classifier-head promotion must not flip a specialist's
    ACTIVE pointer and vice-versa.
    """
    store = _store(tmp_path)
    store.write_candidate("classifier-head", "20260101T000000Z", {"a.bin": b"ch"})
    store.write_candidate("specialist:7", "20260101T000000Z", {"adapter.bin": b"sp"})
    store.promote("classifier-head", "20260101T000000Z")
    assert store.active_version("classifier-head") == "20260101T000000Z"
    assert store.active_version("specialist:7") is None  # untouched


# -------- atomic write helpers --------


def test_atomic_write_bytes_creates_parents(tmp_path):
    target = tmp_path / "deep" / "tree" / "out.bin"
    _atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_atomic_write_text_creates_parents_and_is_utf8(tmp_path):
    target = tmp_path / "deep" / "out.txt"
    _atomic_write_text(target, "café\n")
    assert target.read_text(encoding="utf-8") == "café\n"


def test_atomic_write_bytes_overwrites_existing(tmp_path):
    target = tmp_path / "out.bin"
    target.write_bytes(b"original")
    _atomic_write_bytes(target, b"replaced")
    assert target.read_bytes() == b"replaced"
