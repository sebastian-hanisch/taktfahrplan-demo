"""Fasst die Rohdaten aus tools/raw/ (von tools/sweep.py) zu data/tkt_results.json zusammen.

  python tools/dump_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tkt_constants as C  # noqa: E402

RAW = ROOT / "tools" / "raw"


def load(name):
    p = RAW / name
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


def main():
    res = {"meta": {"search_restarts": C.SEARCH_RESTARTS, "phase_restarts": C.PHASE_RESTARTS, "curve_restarts": C.CURVE_RESTARTS_SWEEP,
                    "random_samples": C.RANDOM_SAMPLES, "n_stops": C.N_STOPS, "T": C.T, "exact_time_limit": C.EXACT_TIME_LIMIT,
                    "seed_start": {str(k): v for k, v in C.SWEEP_SEED_START.items()}, "nets_per_size": C.SWEEP_NETS}, "nets": {}}
    for n in C.SWEEP_LINES:
        rows = [r for r in load(f"nets_{n}.jsonl") if "skip" not in r]
        assert len(rows) == C.SWEEP_NETS, (n, len(rows))
        for r in rows:
            r.pop("wall", None)
            r.pop("t_core", None)
            for h in r["hw"].values():
                h.pop("time", None)
        res["nets"][str(n)] = rows
    res["cp"] = load("cp.jsonl")
    res["dwell"] = load("dwell.jsonl")
    noise = load("noise.jsonl")
    res["noise"] = {}
    for key in ("split", "add"):
        xs = [r[f"{key}_spread"] for r in noise]
        res["noise"][key] = {"n": len(xs), "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)), "max": max(xs), "gt005": sum(1 for x in xs if x > 0.05)}
    rc = json.loads((RAW / "recip.json").read_text(encoding="utf-8"))
    res["recip"] = {"pairs": rc["pairs"], "instances": rc["instances"], "violations": rc["violations"], "not_attainable": rc["not_attainable"],
                    "two_values_wrong": rc["two_values_wrong"], "low_share": rc["low_offsets"] / rc["total_offsets"],
                    "mean_c": rc["c_sum"] / rc["pairs"], "c_zero": rc["c_zero"]}
    out = ROOT / C.RESULTS_FILE
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    print("geschrieben:", out, round(out.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
