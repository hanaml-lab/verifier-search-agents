import importlib.util
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_toy_pipeline_runs_fast_and_beats_pass1():
    spec = importlib.util.spec_from_file_location("run_toy", ROOT / "scripts" / "run_toy.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = tomllib.loads((ROOT / "configs" / "toy.toml").read_text())
    t0 = time.time()
    res = mod.run(cfg)
    assert time.time() - t0 < 60
    assert res["adaptive"] > res["pass@1"] + 0.1
    assert res["adaptive_tokens"] < res["bok_tokens"]
    assert res["ece_calibrated_on_cal"] <= res["ece_raw_on_cal"] + 1e-9
    assert res["false_finish_rate"] < 0.2
