"""Live-Rechnung, Meldungslogik und Kopplungs-Abschnitt auf kleinen echten Netzen (3 und 5 Linien, schnell)."""
import pytest

import tkt_constants as C
import tkt_evaluation as E
from tkt_network import build_valid_instance


@pytest.fixture(scope="module")
def run3():
    return E.run_live(3, 7, 0, 2)


@pytest.fixture(scope="module")
def run5_hw():
    return E.run_live(5, 11, 4, 2)


def test_stage_keys_labels_and_basic_ordering(run3):
    st = run3["stages"]
    assert tuple(st) == E.STAGE_ORDER and set(E.STAGE_LABELS) == set(E.STAGE_ORDER)
    assert st["add"]["avg"] <= st["split"]["avg"] + 1e-9               # die Zusatzzug-Stufe startet von der Aufteilung (Warmstart)
    assert st["add"]["fleet"] >= st["split"]["fleet"] >= run3["base_fleet"] == st["same"]["fleet"]
    assert all(0 <= s["le10"] <= 1 for s in st.values()) and st["random"]["violations"] is None
    assert run3["lower_bound"] >= C.TRANSFER_MIN and st["phase"]["avg"] >= run3["lower_bound"] - 1e-9     # Phasen-Modell nie unter der Untergrenze
    assert st["random"]["avg"] > st["phase"]["avg"] and st["same"]["avg"] > st["phase"]["avg"]


def test_run_is_deterministic():
    a, b = E.run_live(3, 21, 0, 1), E.run_live(3, 21, 0, 1)
    assert {k: v["avg"] for k, v in a["stages"].items()} == {k: v["avg"] for k, v in b["stages"].items()}
    assert [e["avg"] for e in a["curve"]] == [e["avg"] for e in b["curve"]]


def test_budget_zero_gives_the_split_stage_and_budget_changes_the_add_stage():
    zero, two = E.run_live(3, 7, 0, 0), E.run_live(3, 7, 0, 2)
    assert abs(zero["stages"]["add"]["avg"] - zero["stages"]["split"]["avg"]) < 1e-9 and zero["stages"]["add"]["fleet"] == zero["stages"]["split"]["fleet"]
    assert two["stages"]["add"]["avg"] < zero["stages"]["add"]["avg"] and two["stages"]["add"]["fleet"] > zero["stages"]["add"]["fleet"]


def test_headway_run_respects_the_condition_and_reports_references(run5_hw):
    st = run5_hw["stages"]
    for k in ("phase", "split", "add"):
        assert st[k]["violations"] == 0, k
    assert run5_hw["free_split"] is not None and run5_hw["headway"] == 4
    assert st["split"]["avg"] >= run5_hw["free_split"]["avg"] - 0.5      # die Bedingung macht nichts besser (bis aufs Rauschen der Suche)
    assert st["same"]["violations"] > 0                                  # die Referenz verletzt die Zugfolge (sonst wäre sie wirkungslos)


def test_without_headway_there_is_no_free_comparison(run3):
    assert run3["free_split"] is None and E.headway_message(run3["stages"], None, 0, 1.0) is None
    assert all(s["violations"] == 0 for k, s in run3["stages"].items() if k != "random")


def test_three_message_states_at_the_noise_threshold():
    assert E.verdict(1.5, 1.0, "a", "b") == ("over", "a")
    assert E.verdict(0.5, 1.0, "a", "b") == ("noise", "b")
    assert E.verdict(0.0, 1.0, "a", "b") == ("none", "b") and E.verdict(-2.0, 1.0, "a", "b")[0] == "none"
    assert E.verdict(1.0, 1.0, "a", "b")[0] == "noise"                   # genau auf der Schwelle gilt noch als Rauschen
    fake = {"phase": {"avg": 20.0}, "split": {"avg": 17.0}}
    assert E.split_message(fake, 0.95)[1][0] == "over" and E.split_message(fake, 5.0)[1][0] == "noise"
    free = {"avg": 16.0}
    assert E.headway_message(fake, free, 4, 0.95)[1] == "over" and E.headway_message(fake, free, 4, 5.0)[1] == "noise"
    assert E.headway_message(fake, {"avg": 18.0}, 4, 0.95)[1] == "none"


def test_pair_curves_take_two_values_and_the_low_value_c_plus_one_times():
    for n_lines, seed in ((3, 7), (5, 11), (6, 3)):
        inst, _, _, _ = build_valid_instance(n_lines, seed)
        pairs = E.sorted_pairs(inst)
        assert pairs and all(pairs[i][2] >= pairs[i + 1][2] for i in range(len(pairs) - 1))
        for pair in pairs[:6]:
            hin, rueck, sums, c, low, n_low = E.pair_curves(inst, pair)
            assert set(sums) <= {low, low + inst.T} and low == 2 * inst.transfer_min + c and n_low == c + 1 and len(hin) == len(rueck) == inst.T
            assert all(a >= inst.transfer_min and b >= inst.transfer_min for a, b in zip(hin, rueck))


def test_shifting_the_standing_time_moves_the_lower_sum():
    inst, _, _, _ = build_valid_instance(5, 11)
    seen = set()
    for pair in E.sorted_pairs(inst)[:10]:
        lo, hi, std = E.pair_shift_range(inst, pair)
        if hi > lo:
            lows = {E.pair_curves(inst, pair, (s, 0))[4] for s in range(lo, hi + 1)}
            seen.add(len(lows) > 1)
    assert True in seen, "mindestens ein Anschluss: die Aufteilung der Standzeit ändert die kleinste Summe (sonst bräche sie die Kopplung nie)"


def test_top_pairs_table_sums_respect_the_lower_bound_for_standard_splits(run3):
    inst = run3["inst"]
    rows = E.top_pairs_table(inst, run3["stages"]["phase"])
    assert rows and all(r["Summe (min)"] >= r["Untergrenze der Summe (Standard-Standzeit)"] for r in rows)
    assert all(r["Summe (min)"] == r["Wartezeit hin (min)"] + r["Wartezeit zurück (min)"] for r in rows)


def test_view_schedule_has_minutes_in_the_period_and_terminals_without_departure(run3):
    inst = run3["inst"]
    sched = E.view_schedule(inst, run3["stages"]["phase"])
    assert len(sched) == len(inst.lines)
    for li, rows in sched:
        assert rows[0][2] is None and rows[-1][1] is None                  # am ersten Halt fährt nichts zurück ab, am letzten nichts hin
        assert all(0 <= v < inst.T for _, h, r in rows for v in (h, r) if v is not None)


def test_exact_run_matches_a_small_instance_and_reports_a_bound():
    r = E.run_exact(3, 7, 0, 0, 10.0)
    assert r["status"] in ("OPTIMAL", "FEASIBLE") and r["bound"] <= r["avg"] + 1e-6
    live = E.run_live(3, 7, 0, 0)
    assert abs(live["stages"]["phase"]["avg"] - r["avg"]) < 1e-6 or r["status"] == "FEASIBLE"      # bei bewiesenem Optimum = Lokalsuche auf 3 Linien
    assert r["seed"] == live["seed"]
