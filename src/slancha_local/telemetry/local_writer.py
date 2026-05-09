"""Append JSONL trace writer with consent gate."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from slancha_local.telemetry.schema import Trace

logger = logging.getLogger(__name__)


class LocalTraceWriter:
    def __init__(self, root: Path | str) -> None:
        # Check forbidden prefixes BEFORE resolve(), since macOS /etc → /private/etc
        # symlink resolution would defeat the check.
        raw = Path(root).expanduser()
        forbidden_prefixes = ("/etc", "/var", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc")
        raw_str = str(raw)
        if any(raw_str == p or raw_str.startswith(p + "/") for p in forbidden_prefixes):
            raise ValueError(f"refusing to write traces under system path: {raw}")
        root_resolved = raw.resolve()
        root_resolved.mkdir(parents=True, exist_ok=True)
        self._root = root_resolved

    @property
    def root(self) -> Path:
        return self._root

    def _today_path(self) -> Path:
        d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._root / f"{d}.jsonl"

    def write(self, trace: Trace) -> None:
        if not trace.consent_at_capture:
            trace = trace.model_copy(update={"prompt": None, "response": None})
        line = trace.model_dump_json() + "\n"
        path = self._today_path()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.error("trace write failed: %s", e)
