"""Statistik der Messreihe an einer von Hand gerechneten Mini-Ergebnisdatei (zwei Netzgrößen, je zwei Netze)."""
import math

import pytest

import tkt_results as R


def rec(phase, split, add, greedy, lb, hw4_split, viol=1, segs=4, fleet=(10, 12), curve=((10, 20.0), (11, 18.0), (12, 17.0))):
    stage = lambda avg, le, fl=None: {"avg": avg, "le10": le, **({"fleet": fl} if fl else {})}
    return {"seed": 1, "n_lines": 3, "base_fleet": 10, "rank": 1, "transfer_share": 0.1, "n_segment_pairs": segs,
            "same": stage(36.0, 0.1), "random": stage(32.5, 0.1), "greedy": stage(greedy, 0.3), "phase": stage(phase, 0.3),
            "split": stage(split, 0.5, fleet[0]), "add": stage(add, 0.7, fleet[1]), "lower_bound": lb,
            "curve": [{"avg": a, "fleet": f} for f, a in curve],
            "hw": {str(h): {"split": {"avg": hw4_split, "feasible": True}, "add": {"avg": hw4_split - 1, "feasible": True}} for h in (3, 4, 5)},
            "viol_free": {str(h): viol for h in (3, 4, 5)}}


@pytest.fixture
def res():
    return {"nets": {"3": [rec(24.0, 18.0, 15.0, 25.0, 18.0, 19.0), rec(26.0, 20.0, 16.0, 26.0, 20.0, 21.0)],
                      "5": [rec(25.0, 19.0, 15.0, 26.0, 19.0, 20.0, viol=2), rec(23.0, 17.0, 14.0, 24.0, 17.0, 18.0, viol=0)]},
            "noise": {"split": {"p95": 0.95, "median": 0.1, "max": 1.8, "n": 30}, "add": {"p95": 0.57, "median": 0.27, "max": 1.0, "n": 30}}}


def test_mean_se_and_describe():
    m, se = R.mean_se([2.0, 4.0, 6.0])
    assert m == 4.0 and abs(se - 2.0 / math.sqrt(3)) < 1e-12
    d = R.describe(list(range(1, 11)))
    assert d["n"] == 10 and d["mean"] == 5.5 and d["median"] == 5.5 and abs(d["p10"] - 1.9) < 1e-9 and abs(d["p90"] - 9.1) < 1e-9


def test_cascade_means_and_stage_order(res):
    c = R.cascade(res, 3)
    assert c["phase"]["mean"] == 25.0 and c["split"]["mean"] == 19.0 and c["add"]["mean"] == 15.5 and c["greedy"]["mean"] == 25.5
    assert c["split"]["fleet"] == 10 and c["add"]["fleet"] == 12 and c["same"]["fleet"] == 10          # Referenzstufen: Basisbestand
    assert abs(c["phase"]["le10"] - 0.3) < 1e-12 and c["lower_bound"]["mean"] == 19.0
    assert [k for k in R.STAGES] == ["same", "random", "greedy", "phase", "split", "add"]
    pooled = R.cascade(res)
    assert pooled["phase"]["n"] == 4 and pooled["phase"]["mean"] == 24.5


def test_paired_differences_and_coupling_share(res):
    p = R.paired(res, 3)
    assert p["split_gain"]["mean"] == 6.0 and p["add_gain"]["mean"] == 3.5 and p["headway_cost"]["mean"] == 1.0 and p["greedy_gap"]["mean"] == 0.5
    assert p["positive_split"] == 2 and p["n"] == 2 and p["min_split_gain"] == 6.0
    # (lb - 3) / (phase - 3): (15/21 + 17/23) / 2
    assert abs(R.coupling_share(res, 3) - (15 / 21 + 17 / 23) / 2) < 1e-12


def test_size_curves_headway_rows_and_curve_means(res):
    sc = R.size_curves(res)
    assert sc["phase"] == {3: 25.0, 5: 24.0} and sc["split"][5] == 18.0 and sc["add"][3] == 15.5
    rows = R.headway_rows(res)
    assert len(rows) == 6 and {r["h"] for r in rows} == {3, 4, 5}
    r5 = next(r for r in rows if r["lines"] == 5 and r["h"] == 4)
    assert r5["cost"] == 1.0 and r5["feasible"] == 2 and r5["nets_violated"] == 1 and abs(r5["pair_share"] - (2 / 4 + 0) / 2 * 100) < 1e-9
    cm = R.curve_means(res, 3)
    assert [(k, f, a) for k, f, a, _ in cm] == [(0, 0.0, 20.0), (1, 1.0, 18.0), (2, 2.0, 17.0)]
    assert R.add_stage_at(res, 3, 1)["mean"] == 18.0 and R.add_stage_at(res, 3, 99)["mean"] == 17.0      # Schritt über das Ende: letzter Wert


def test_noise_thresholds_and_nearest_lines(res):
    assert R.noise_thresholds(res) == {"split": 0.95, "add": 0.57}
    assert R.nearest_lines(7) == 6 and R.nearest_lines(4) == 3 and R.nearest_lines(100) == 10 and R.nearest_lines(0) == 3
