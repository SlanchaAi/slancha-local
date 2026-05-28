"""CLI tests for ``slancha gate-decide``."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slancha_local.cli import app
from slancha_local.train.eval_row import EvalSample, aggregate_eval_pass

runner = CliRunner()


def _write_row(path: Path, samples: list[EvalSample], **kw) -> None:
    row = aggregate_eval_pass(samples, **kw)
    path.write_text(json.dumps(row))


def _samples(*, domain="general", model="m", score=4.0, n=200) -> list[EvalSample]:
    return [EvalSample(domain=domain, served_model=model, score=score) for _ in range(n)]


def test_gate_decide_accept_exits_zero(tmp_path):
    champ = tmp_path / "champ.json"
    chall = tmp_path / "chall.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=4.0),
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    res = runner.invoke(app, ["gate-decide", "--champion", str(champ), "--challenger", str(chall)])
    assert res.exit_code == 0, res.output
    assert "ACCEPT" in res.output
    assert "v1" in res.output and "v2" in res.output


def test_gate_decide_reject_exits_two(tmp_path):
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=3.01),  # +0.01 < default 0.05
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    res = runner.invoke(app, ["gate-decide", "--champion", str(champ), "--challenger", str(chall)])
    assert res.exit_code == 2
    assert "REJECT" in res.output
    assert "mean_delta" in res.output


def test_gate_decide_json_output_is_parseable(tmp_path):
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=4.0),
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion",
            str(champ),
            "--challenger",
            str(chall),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert payload["accept"] is True
    assert payload["champion_version"] == "v1"
    assert payload["challenger_version"] == "v2"


def test_gate_decide_threshold_overrides_accept_marginal(tmp_path):
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=3.02),  # +0.02 — passes when --mean-delta 0.01
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion",
            str(champ),
            "--challenger",
            str(chall),
            "--mean-delta",
            "0.01",
        ],
    )
    assert res.exit_code == 0
    assert "ACCEPT" in res.output


def test_gate_decide_no_require_judge_match_allows_different_judges(tmp_path):
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0, domain="g"),
        router_version="v1",
        judge_model="judge-a",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=4.0, domain="g"),
        router_version="v2",
        judge_model="judge-b",
        holdout_version=1,
        ts="t2",
    )
    res_reject = runner.invoke(app, ["gate-decide", "--champion", str(champ), "--challenger", str(chall)])
    assert res_reject.exit_code == 2
    assert "judge_model" in res_reject.output

    res_accept = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion",
            str(champ),
            "--challenger",
            str(chall),
            "--no-require-judge-match",
        ],
    )
    assert res_accept.exit_code == 0
    assert "ACCEPT" in res_accept.output


def test_gate_decide_missing_file_exits_one(tmp_path):
    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion",
            str(tmp_path / "nope.json"),
            "--challenger",
            str(tmp_path / "also-nope.json"),
        ],
    )
    assert res.exit_code == 1


def test_gate_decide_reads_jsonl_last_row(tmp_path):
    """A two-line JSONL file should be read as 'the most recent row'.
    Mimics how mesh's append_pass / our append_eval_row would lay things out."""
    champ = tmp_path / "champ.jsonl"
    chall = tmp_path / "chall.jsonl"
    # First row is stale; second is current
    champ_stale = aggregate_eval_pass(
        _samples(score=4.5),  # high score would auto-pass
        router_version="vOLD",
        judge_model="j",
        holdout_version=1,
        ts="t0",
    )
    champ_cur = aggregate_eval_pass(
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    champ.write_text(json.dumps(champ_stale) + "\n" + json.dumps(champ_cur) + "\n")
    chall_cur = aggregate_eval_pass(
        _samples(score=4.0),
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    chall.write_text(json.dumps(chall_cur) + "\n")
    res = runner.invoke(
        app,
        ["gate-decide", "--champion", str(champ), "--challenger", str(chall), "--json"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert payload["champion_version"] == "v1"  # second row, not vOLD


def _verdict_paths(tmp_path: Path) -> tuple[Path, Path]:
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=4.0),
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    return champ, chall


def test_promotions_log_writes_verdict_when_flag_set(tmp_path: Path):
    """``--promotions-log`` event-sources the verdict to the named JSONL."""
    champ, chall = _verdict_paths(tmp_path)
    log = tmp_path / "dashboard" / "promotions.jsonl"
    assert not log.exists()

    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion", str(champ),
            "--challenger", str(chall),
            "--promotions-log", str(log),
        ],
    )

    assert res.exit_code == 0, res.output
    assert log.exists(), "promotions log was not written"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["accept"] is True
    assert rows[0]["challenger_version"] == "v2"


def test_promotions_log_appends_on_repeat_invocations(tmp_path: Path):
    """Two CLI runs must append (not truncate) — the log is event-sourced."""
    champ, chall = _verdict_paths(tmp_path)
    log = tmp_path / "p.jsonl"

    for _ in range(2):
        res = runner.invoke(
            app,
            [
                "gate-decide",
                "--champion", str(champ),
                "--challenger", str(chall),
                "--promotions-log", str(log),
            ],
        )
        assert res.exit_code == 0

    rows = log.read_text().splitlines()
    assert len(rows) == 2


def test_promotions_log_records_reject_too(tmp_path: Path):
    """Reject verdicts must also be event-sourced — that's the audit point."""
    champ = tmp_path / "c.json"
    chall = tmp_path / "x.json"
    _write_row(
        champ,
        _samples(score=3.0),
        router_version="v1",
        judge_model="j",
        holdout_version=1,
        ts="t1",
    )
    _write_row(
        chall,
        _samples(score=3.01),  # < default mean-delta
        router_version="v2",
        judge_model="j",
        holdout_version=1,
        ts="t2",
    )
    log = tmp_path / "p.jsonl"

    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion", str(champ),
            "--challenger", str(chall),
            "--promotions-log", str(log),
        ],
    )

    assert res.exit_code == 2
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["accept"] is False
    assert rows[0]["reject_reasons"]


def test_promotions_log_omitted_writes_nothing(tmp_path: Path):
    """No ``--promotions-log`` → no log pollution. Useful for dry-runs."""
    champ, chall = _verdict_paths(tmp_path)
    res = runner.invoke(
        app,
        [
            "gate-decide",
            "--champion", str(champ),
            "--challenger", str(chall),
        ],
    )
    assert res.exit_code == 0
    assert not any(tmp_path.glob("**/*.jsonl")), "verdict was logged despite no --promotions-log"
