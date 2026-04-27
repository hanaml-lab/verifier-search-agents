from ccv.render import execution_features, render_prefix


def test_execution_features_pick_last_test_run():
    steps = [{"type": "cmd", "cmd": "ls", "exit": 0, "output": "a"},
             {"type": "cmd", "cmd": "pytest tests/x.py", "exit": 1, "output": "...\n1 failed"},
             {"type": "edit", "path": "x.py"},
             {"type": "cmd", "cmd": "python -m pytest -q", "exit": 0, "output": "3 passed\n"}]
    f = execution_features(steps)
    assert f == {"n_test_runs": 2, "last_exit": 0, "last_summary": "3 passed"}
    assert "runs=2" in render_prefix("fix bug", steps, "diff")
