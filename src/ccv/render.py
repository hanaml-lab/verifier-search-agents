"""Render an agent trajectory prefix into verifier input (Section 3.2)."""
from __future__ import annotations

import re

TEST_CMD = re.compile(r"\b(pytest|tox|nox|make test|npm test|go test|cargo test|python -m unittest)\b")


def execution_features(steps: list[dict]) -> dict:
    """Summarise test/lint/build commands the agent ran: count, last exit code, last summary line."""
    runs = [s for s in steps if s.get("type") == "cmd" and TEST_CMD.search(s.get("cmd", ""))]
    last = runs[-1] if runs else None
    summary = ""
    if last:
        lines = [l for l in last.get("output", "").splitlines() if l.strip()]
        summary = lines[-1] if lines else ""
    return {"n_test_runs": len(runs), "last_exit": None if last is None else last.get("exit", None), "last_summary": summary}


def render_prefix(instruction: str, steps: list[dict], diff: str, obs_chars: int = 2000) -> str:
    feats = execution_features(steps)
    log = []
    for s in steps:
        if s.get("type") == "cmd":
            log.append(f"$ {s['cmd']}\n[exit {s.get('exit')}] {s.get('output', '')[-obs_chars:]}")
        elif s.get("type") == "edit":
            log.append(f"[edit] {s.get('path')}")
    return (f"### Task\n{instruction}\n\n### Execution\nruns={feats['n_test_runs']} last_exit={feats['last_exit']} "
            f"summary={feats['last_summary']}\n\n### Diff\n{diff}\n\n### Log\n" + "\n".join(log))
