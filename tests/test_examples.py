import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrate_threshold_on_the_bundled_scores():
    out = _example("calibrate_threshold").main([str(ROOT / "examples" / "data" / "verifier_scores.csv"), "--alpha", "0.10"])
    assert 0.0 < out["threshold"] < 1.0
    decisions = [d["decision"] for d in out["decisions"]]
    assert len(decisions) == 8 and "submit" in decisions and "hold back" in decisions
    assert all((d["q"] >= out["threshold"]) == (d["decision"] == "submit") for d in out["decisions"])
