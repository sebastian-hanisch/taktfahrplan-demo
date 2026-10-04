"""Jede Zahl aus dem README wird hier aus data/tkt_results.json nachgerechnet; `CLAIMS` nennt den Text, der im README stehen muss.

Die Zahlen der Messreihe sind Mittel über je 30 Netze (150 insgesamt) und kommen aus tools/sweep.py (Lokalsuche mit 40 Neustarts);
Ordnungen und Schranken sind exakt geprüft, Mittelwerte mit gerundeter Anzeige.
"""
from pathlib import Path

import pytest

import tkt_constants as C
import tkt_results as R
import tkt_stories as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def res():
    return R.load_results()


def f1(x):
    return f"{x:.1f}"


def test_every_net_size_has_thirty_nets_and_the_meta_matches_the_constants(res):
    assert {int(k): len(v) for k, v in res["nets"].items()} == {n: C.SWEEP_NETS for n in C.SWEEP_LINES}
    m = res["meta"]
    assert m["search_restarts"] == C.SEARCH_RESTARTS and m["T"] == C.T and m["n_stops"] == C.N_STOPS
    assert all(m["seed_start"][str(n)] == C.SWEEP_SEED_START[n] for n in C.SWEEP_LINES)
    assert len(R.nets(res)) == 150 and "150 Netze" in README


def test_cascade_order_for_every_net_size(res):
    for n in C.SWEEP_LINES:
        c = R.cascade(res, n)
        means = [c[k]["mean"] for k in ("same", "random", "greedy", "phase", "split", "add")]
        assert means[3] > means[4] > means[5], (n, means)                      # jede Stufe nach Phasen bringt etwas
        assert means[1] > means[3] and means[0] > means[3] and means[2] >= means[3] - 1e-9
        assert c["lower_bound"]["mean"] <= c["phase"]["mean"] + 1e-9
        assert 3.0 < c["random"]["mean"] < 40 and abs(c["random"]["mean"] - (3 + (C.T - 1) / 2)) < 1.0      # Erwartung zufälliger Phasen: 3 + (T-1)/2


def test_split_gain_is_positive_in_every_net_and_the_lower_bound_is_never_beaten_by_phases(res):
    rows = R.nets(res)
    assert all(r["phase"]["avg"] - r["split"]["avg"] > 0 for r in rows)
    assert all(r["phase"]["avg"] >= r["lower_bound"] - 1e-9 for r in rows)
    pr = R.paired(res)
    assert pr["positive_split"] == pr["n"] == 150
    assert f"{pr['positive_split']}/{pr['n']}" in README or "in allen 150 Netzen" in README


def test_split_gain_and_coupling_share_in_the_readme(res):
    pr = R.paired(res)
    sg = pr["split_gain"]
    assert f"{f1(sg['mean'])} ± {f1(sg['se'])} min" in README, f"{f1(sg['mean'])} ± {f1(sg['se'])}"
    assert f"{R.coupling_share(res) * 100:.0f} %" in README


def test_cascade_means_for_six_lines_in_the_readme(res):
    c = R.cascade(res, 6)
    for k in ("random", "phase", "split"):
        assert f1(c[k]["mean"]) in README, (k, f1(c[k]["mean"]))
    assert f"{100 * c['phase']['le10']:.0f} %" in README and f"{100 * c['split']['le10']:.0f} %" in README


def test_density_is_not_the_problem(res):
    phase = R.size_curves(res)["phase"]
    assert max(phase.values()) - min(phase.values()) < 1.5, phase
    nets = R.nets(res)
    xs = [r["rank"] for r in nets]
    ys = [r["phase"]["avg"] for r in nets]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    r_val = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) ** 0.5 * sum((y - my) ** 2 for y in ys) ** 0.5)
    assert abs(r_val) < 0.3, r_val                                              # Kreisrang erklärt die Wartezeit kaum
    assert f"{r_val:.2f}".replace("-", "−") in README or f"{r_val:.2f}" in README, f"{r_val:.2f}"


def test_headway_costs_grow_with_the_number_of_lines_and_the_stage(res):
    rows = R.headway_rows(res)
    by = {(r["lines"], r["h"]): r["cost"] for r in rows}
    assert by[(10, 4)] > by[(3, 4)] and all(by[(n, 5)] >= by[(n, 3)] - 1e-9 for n in C.SWEEP_LINES)
    assert all(r["feasible"] == r["n"] for r in rows)                           # überall ein zulässiger Fahrplan gefunden
    assert rows[0]["nets_violated"] <= rows[0]["n"]
    assert f1(R.paired(res)["headway_cost"]["mean"]) in README


def test_additional_trains_saturate(res):
    cm = R.curve_means(res, 6)
    gains = [a[2] - b[2] for a, b in zip(cm, cm[1:])]
    assert all(g >= -1e-9 for g in gains) and gains[0] > gains[-1] and gains[-1] <= 0.2, gains
    assert f1(R.paired(res)["add_gain"]["mean"]) in README


def test_greedy_is_close_to_the_phase_optimum_and_cp_sat_proves_little(res):
    gap = R.paired(res)["greedy_gap"]
    assert 0 <= gap["mean"] < 1.5
    cps = R.cp_summary(res)
    assert cps["proven"] <= cps["n"] // 2                                       # beim erweiterten Modell beweist CP-SAT wenig
    assert cps["ls40_mean"] <= cps["ls20_mean"] + 0.05 and cps["ls40_max_dev"] < cps["ls20_max_dev"] + 1e-9
    assert f"{cps['proven']}/{cps['n']}" in README or f"{cps['proven']} von {cps['n']}" in README


def test_recip_check_found_no_violation(res):
    rc = res["recip"]
    assert rc["violations"] == 0 and rc["not_attainable"] == 0 and rc["two_values_wrong"] == 0 and rc["pairs"] >= 300
    assert str(rc["pairs"]) in README


def test_noise_thresholds_are_in_the_expected_range(res):
    th = R.noise_thresholds(res)
    assert 0.3 < th["split"] < 2.0 and 0.2 < th["add"] < 2.0
    assert f"{th['split']:.2f}" in README


def test_dwell_flexibility_helps_in_the_exact_model_only(res):
    d = R.dwell_summary(res)
    assert d["dwell0"] < d["split0"] and d["dwell1"] < d["dwell0"] and d["n"] == 15


def test_transfer_share_is_small(res):
    shares = [r["transfer_share"] for r in R.nets(res)]
    assert 0.03 < min(shares) and max(shares) < 0.30, (min(shares), max(shares))
    assert f"{100 * min(shares):.0f}" in README and f"{100 * max(shares):.0f}" in README


def test_late_share_with_extra_trains_at_six_lines_is_over_a_third(res):
    late = 100 - 100 * R.cascade(res, 6)["add"]["le10"]
    assert late > 100 / 3
    assert f"{late:.0f} %" in README


@pytest.mark.parametrize("name", C.PRESET_ORDER)
def test_every_preset_tells_its_story_on_the_shown_network(res, name):
    """Abnahmekriterien aus tkt_stories an dem Netz, das der Preset-Knopf tatsächlich zeigt, und an der Messreihe."""
    import tkt_evaluation as E
    p = C.PRESETS[name]
    run = E.run_live(p["lines"], p["seed"], p["headway"], p["budget"])
    facts = S.facts_for(run, res, R.noise_thresholds(res))
    failed = [text for _, text, ok in S.check(name, facts) if not ok]
    assert not failed, (name, failed, {k: round(v, 3) if isinstance(v, float) else v for k, v in facts.items()})
