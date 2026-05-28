"""Unit tests for ``slancha_local.train.gate.append_verdict``.

The function is a near-trivial mirror of ``mesh.eval.gate.append_verdict``,
but the contract carries SRE-level expectations: every promotion event must
survive a fresh-deployment ``dashboard/`` directory not existing, must be
append-only (never rewrite a row), and must JSON-encode in a shape any
downstream reader can ingest. These tests pin those properties so a future
refactor can't break them silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from slancha_local.train.gate import PromotionVerdict, append_verdict


def _verdict(accept: bool = True, version: str = "v2") -> PromotionVerdict:
    return PromotionVerdict(
        accept=accept,
        reject_reasons=() if accept else ("test reason",),
        mean_delta=0.07,
        per_domain_deltas={"general": 0.05, "code": 0.10},
        champion_version="v1",
        challenger_version=version,
        n_eval_champion=200,
        n_eval_challenger=200,
        judge_model_champion="j",
        judge_model_challenger="j",
        decided_at="2025-01-01T00:00:00Z",
        thresholds={"mean_score_delta": 0.05},
    )


def test_append_verdict_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """A fresh deployment with no dashboard/ directory must still log."""
    log = tmp_path / "dashboard" / "promotions.jsonl"
    assert not log.parent.exists()

    append_verdict(log, _verdict())

    assert log.exists(), "promotions log not written"
    line = log.read_text().strip()
    row = json.loads(line)
    assert row["accept"] is True
    assert row["challenger_version"] == "v2"


def test_append_verdict_is_append_only_across_calls(tmp_path: Path) -> None:
    """Two calls must produce two JSONL rows in order — never rewrite."""
    log = tmp_path / "promotions.jsonl"
    append_verdict(log, _verdict(accept=True, version="v2"))
    append_verdict(log, _verdict(accept=False, version="v3"))

    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 2
    assert [r["challenger_version"] for r in rows] == ["v2", "v3"]
    assert [r["accept"] for r in rows] == [True, False]


def test_append_verdict_serializes_reject_reasons_as_list(tmp_path: Path) -> None:
    """``reject_reasons`` is stored as a tuple on the dataclass but must
    serialize as a JSON list (tuples aren't a JSON type; readers parse
    them back as lists). Mirrors mesh's contract."""
    log = tmp_path / "p.jsonl"
    append_verdict(log, _verdict(accept=False))
    row = json.loads(log.read_text().strip())
    assert isinstance(row["reject_reasons"], list)
    assert row["reject_reasons"] == ["test reason"]


def test_append_verdict_round_trip_via_eval_row_reader_shape(tmp_path: Path) -> None:
    """The written line is plain JSON — `json.loads` of any single line
    yields the same dict as `verdict.to_row()`."""
    log = tmp_path / "p.jsonl"
    v = _verdict()
    append_verdict(log, v)
    parsed = json.loads(log.read_text().strip())
    assert parsed == v.to_row()
