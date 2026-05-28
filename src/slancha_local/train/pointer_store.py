"""Versioned artifact pointer-store with atomic ACTIVE-pointer promotion.

Built for P2b.3 (classifier-head retrain loop) and forward-designed so the
same primitive carries P3's per-cluster LLM specialist adapters.

Layout:

    <root>/
      <component>/
        ACTIVE              one-line file naming the live version
        <version>/          arbitrary files written by the writer
        <version>/
        ...

Why a *file* instead of a symlink? Two reasons:

* Portability — Windows / network filesystems / some container layers don't
  guarantee symlink atomicity; an atomic ``os.replace`` of a regular file
  works everywhere POSIX-rename does.
* Consistency with the rest of the codebase — we already use
  tmpfile + fsync + rename for the P2a snapshot writer, so the
  durability story is uniform.

Promotion is atomic: the ACTIVE pointer flips from one version to another
via tmpfile + rename in a single filesystem op. A crash mid-write leaves
ACTIVE pointing at whichever side wins the rename — never partial.

Rollback is instant: as long as the prior version's directory is still
on disk (governed by ``keep_versions``), pointing ACTIVE back is one
rename.

Loaders should follow this resolution order:

1. ``active_path(component, filename)`` — ACTIVE points at a version that
   contains ``filename``.
2. fall back to whatever legacy fixed path the component used before this
   store existed (e.g. ``assets/classifier_v1/<bin>``). When no ACTIVE
   pointer exists, the store is effectively inert and the loader sees
   exactly the pre-store behavior — back-compat by default.

This module has zero non-stdlib dependencies on purpose: the pointer
store is a primitive that should be loadable in any context the rest of
slancha-local runs in.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ACTIVE_FILENAME = "ACTIVE"

# Version strings are validated to keep promotions away from path-traversal
# tricks (``..`` / absolute paths / shell metacharacters) when callers pass
# user-supplied strings. Timestamps emitted by ``new_version()`` always
# satisfy this; explicit callers can use any ASCII identifier.
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PointerStoreError(RuntimeError):
    """Base for pointer-store errors (invalid version, missing component, …)."""


def new_version(now: datetime | None = None) -> str:
    """Return a fresh monotonic version string (UTC second-precision)."""
    when = now or datetime.now(UTC)
    return when.strftime("%Y%m%dT%H%M%SZ")


def _validate_version(version: str) -> None:
    if not _VERSION_RE.match(version):
        raise PointerStoreError(
            f"invalid version string {version!r}: must match {_VERSION_RE.pattern}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmpfile + fsync + rename).

    Same recipe the P2a snapshot writer uses, lifted into a helper so the
    promotion path and the candidate writer share one durability story.

    Note: we fsync the file but NOT the parent dir after ``os.replace``.
    A power-loss right after the rename could drop the promotion, but
    that fails SAFE — ACTIVE simply stays at the prior version (or
    falls back to the legacy fixed path), it never silently points at
    a corrupt/torn target. Belt-and-suspenders would be to fsync the
    parent fd; not done here because the failure mode is benign for a
    promotion pointer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (tmpfile + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class PointerStore:
    """Versioned artifact store with an atomic ACTIVE pointer per component.

    Construction is cheap and pure (no filesystem ops); directories are
    created lazily on first write so importing this module has zero
    side-effects on disk.
    """

    def __init__(self, root: Path, *, keep_versions: int = 5) -> None:
        if keep_versions < 1:
            raise PointerStoreError("keep_versions must be >= 1")
        self.root = Path(root)
        self.keep_versions = keep_versions

    # -------- component / version paths --------

    def component_dir(self, component: str) -> Path:
        return self.root / component

    def version_dir(self, component: str, version: str) -> Path:
        _validate_version(version)
        return self.component_dir(component) / version

    def _active_file(self, component: str) -> Path:
        return self.component_dir(component) / ACTIVE_FILENAME

    # -------- read side --------

    def active_version(self, component: str) -> str | None:
        """Return the version named in ACTIVE, or ``None`` if no pointer exists.

        Inert default — when this returns ``None``, callers fall back to
        their pre-store fixed-path artifact (preserving today's behavior
        on first deploy of the loop).
        """
        active = self._active_file(component)
        if not active.exists():
            return None
        version = active.read_text(encoding="utf-8").strip()
        if not version:
            return None
        try:
            _validate_version(version)
        except PointerStoreError:
            logger.warning(
                "ACTIVE pointer for %s contains invalid version %r — ignoring",
                component,
                version,
            )
            return None
        return version

    def active_path(self, component: str, filename: str) -> Path | None:
        """Return the absolute path to ``filename`` in the active version dir.

        Returns ``None`` when no ACTIVE pointer exists, or when the active
        version's directory is missing on disk (corruption case — caller
        falls back to the legacy fixed path).
        """
        version = self.active_version(component)
        if version is None:
            return None
        candidate = self.version_dir(component, version) / filename
        if not candidate.exists():
            logger.warning(
                "ACTIVE pointer for %s references version %s but %s is missing",
                component,
                version,
                candidate,
            )
            return None
        return candidate

    def versions(self, component: str) -> list[str]:
        """Return all version directories that currently exist on disk, sorted."""
        cdir = self.component_dir(component)
        if not cdir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in cdir.iterdir()
            if entry.is_dir() and _VERSION_RE.match(entry.name)
        )

    # -------- write side --------

    def write_candidate(
        self,
        component: str,
        version: str,
        files: dict[str, bytes],
    ) -> Path:
        """Write a candidate version's files to disk (does NOT flip ACTIVE).

        Returns the version directory. Use this for the
        verify-load-before-commit pattern: write candidate → smoke-load →
        only call ``promote()`` after the smoke load succeeds.
        """
        _validate_version(version)
        vdir = self.version_dir(component, version)
        vdir.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            if "/" in name or "\\" in name or name in ("", ".", "..", ACTIVE_FILENAME):
                raise PointerStoreError(
                    f"invalid candidate filename {name!r} for component {component!r}"
                )
            _atomic_write_bytes(vdir / name, payload)
        return vdir

    def promote(self, component: str, version: str) -> None:
        """Atomically flip ACTIVE to ``version`` and prune old versions.

        Refuses to promote a version that doesn't have a directory on disk
        (catches the "called promote without write_candidate" bug early).
        """
        _validate_version(version)
        vdir = self.version_dir(component, version)
        if not vdir.is_dir():
            raise PointerStoreError(
                f"cannot promote {component}/{version}: version dir does not exist"
            )
        _atomic_write_text(self._active_file(component), version + "\n")
        self._prune(component, keep_active=version)

    def rollback(self, component: str) -> str | None:
        """Point ACTIVE back to the most recent version BEFORE the current ACTIVE.

        Walks chronologically: finds the current ACTIVE in the sorted
        version list, takes the slice strictly before it, picks the last
        element of that slice. Repeated calls walk steadily backward
        (v3→v2→v1→None) instead of oscillating to whatever happens to be
        "not current" (which would bounce forward to a newer un-pruned
        version on the second call).

        Returns the version we rolled back to, or ``None`` if there is no
        prior version on disk (in which case ACTIVE is removed entirely
        and the loader falls back to the legacy fixed path).
        """
        current = self.active_version(component)
        versions = self.versions(component)
        if current is not None and current in versions:
            # Chronologically prior — strictly before the current index.
            prior = versions[: versions.index(current)]
        else:
            # ACTIVE missing or pointing at a pruned version: treat every
            # surviving version as "prior" and pick the most recent.
            prior = versions
        if not prior:
            active = self._active_file(component)
            if active.exists():
                active.unlink()
            return None
        target = prior[-1]
        _atomic_write_text(self._active_file(component), target + "\n")
        return target

    # -------- maintenance --------

    def _prune(self, component: str, *, keep_active: str) -> None:
        """Drop oldest version dirs, but never delete the current ACTIVE."""
        versions = self.versions(component)
        # Always retain ``keep_active`` even if it's an old version.
        retain = set(versions[-self.keep_versions:])
        retain.add(keep_active)
        for version in versions:
            if version in retain:
                continue
            vdir = self.version_dir(component, version)
            try:
                for entry in vdir.iterdir():
                    if entry.is_file():
                        entry.unlink()
                vdir.rmdir()
            except OSError as e:
                logger.warning("failed to prune %s/%s: %s", component, version, e)
