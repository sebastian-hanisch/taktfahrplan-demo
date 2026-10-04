"""Zweig-Tests (DEMO-PLAYBOOK Abschnitt 0): jede Stufe weicht auf mindestens einem Netz von der Nachbarstufe ab, sonst läuft sie womöglich nie.
Dazu Warmstart/Kurve auf echten Netzen. Kleine Neustart-Zahlen: nur Ordnungen und Schranken, keine exakten Werte der Heuristik."""
import statistics as st

import tkt_constants as C
from tkt_network import build_instance_int
from tkt_rng import SplitMix64
from tkt_search import (budget_curve, cost_split, local_search_split, phase_search, split_cache, standard_split_indices, violations_split)
from tkt_timetable import shared_segments

SEEDS = range(0, 8)


def net(seed, lines=6):
    inst, _, _ = build_instance_int(C.N_STOPS, lines, C.MAX_LINE_LENGTH, seed)
    return inst


def test_split_stage_differs_from_phase_stage_and_is_better_on_average():
    gains = []
    for seed in SEEDS:
        inst = net(seed)
        tw = sum(inst.transfers.values())
        _, c_ph = phase_search(inst, 10, SplitMix64(seed))
        _, _, c_sp, _ = local_search_split(inst, 10, SplitMix64(seed), [False] * len(inst.lines))
        gains.append((c_ph - c_sp) / tw)
    assert sum(1 for g in gains if g > 1e-6) >= 6, gains          # fast jedes Netz profitiert (Messreihe: 150 von 150)
    assert st.mean(gains) > 1.0, gains


def test_additional_train_stage_differs_from_split_stage():
    better = 0
    for seed in SEEDS:
        inst = net(seed)
        n = len(inst.lines)
        o, si, c, _ = local_search_split(inst, 10, SplitMix64(seed), [False] * n)
        _, _, c2, _ = local_search_split(inst, 10, SplitMix64(seed), [True] * n, init=[(o, si)])
        assert c2 <= c + 1e-9                                         # Warmstart: nie schlechter
        better += c2 < c - 1e-6
    assert better >= 6


def test_headway_changes_the_result_on_dense_nets_and_is_respected():
    changed = 0
    for seed in SEEDS:
        inst = net(seed, 8)
        n = len(inst.lines)
        pairs = shared_segments(inst)
        o, si, c, cache = local_search_split(inst, 10, SplitMix64(seed), [False] * n, hw=5)
        assert c < 1e6 and violations_split(inst, cache, o, si, 5, pairs) == 0     # Bedingung eingehalten (Netze der Messreihe: immer zulässig)
        _, _, c_free, _ = local_search_split(inst, 10, SplitMix64(seed), [False] * n)
        changed += c > c_free + 1e-6
    assert changed >= 4


def test_budget_curve_step_zero_is_the_given_solution_and_curve_is_monotone():
    inst = net(3)
    n = len(inst.lines)
    tw = sum(inst.transfers.values())
    o, si, c, cache = local_search_split(inst, 10, SplitMix64(1), [False] * n)
    curve = budget_curve(inst, n, 1, SplitMix64(2), init=[(o, si)])
    assert abs(curve[0]["avg"] - c / tw) < 1e-9 and curve[0]["offsets"] == o and not any(curve[0]["allowed_lines"])
    avgs = [e["avg"] for e in curve]
    assert all(b <= a + 1e-9 for a, b in zip(avgs, avgs[1:])), avgs
    assert len(curve) == n + 1 and all(sum(e["allowed_lines"]) == e["allowed"] for e in curve)
    fleets = [e["fleet"] for e in curve]
    assert fleets[0] == min(fleets) and fleets[-1] > fleets[0]            # die Erlaubnis wird tatsächlich genutzt
    # jede Kurvenlösung ist ein gültiger Fahrplan: Kosten aus Offsets und Aufteilungen stimmen mit avg überein
    from tkt_timetable import mean_wait, times_for, waits_for
    for e in curve:
        assert abs(mean_wait(waits_for(inst, times_for(inst, e["offsets"], e["splits"]))) - e["avg"]) < 1e-9


def test_standard_split_is_a_candidate_of_every_line():
    inst = net(2)
    cache = split_cache(inst, [False] * len(inst.lines))
    std = standard_split_indices(inst, cache)
    assert len(std) == len(inst.lines) and all(cache[li][0][std[li]][1] == 0 for li in range(len(std)))
    o = [0] * len(inst.lines)
    assert cost_split(inst, cache, o, std) > 0
