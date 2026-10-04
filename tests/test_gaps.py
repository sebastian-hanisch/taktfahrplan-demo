"""Gezielte Tests für Stellen, die der Fehler-Einbau-Test (tools/mutation_check.py) zuerst nicht fand."""
import random

import numpy as np

import tkt_constants as C
from pesp_api import P
from test_pesp_core import mini, shared
from tkt_evaluation import pair_curves, pair_shift_range, sorted_pairs
from tkt_exact import solve_pesp
from tkt_network import build_instance_int, build_valid_instance, demand_greedy
from tkt_rng import SplitMix64
from tkt_timetable import pair_offsets_sum
from tkt_search import budget_curve, descent_split, local_search_split, split_cache, standard_split_indices


def real_net(seed, lines=6):
    return build_instance_int(C.N_STOPS, lines, C.MAX_LINE_LENGTH, seed)[0]


def test_asymmetric_split_moves_the_return_departure_by_hand():
    # L0 der Mini-Instanz: Schlupf 4; Standzeit 0 am ersten Endpunkt: dep_b2 = 8 + 2 = 10, am zweiten 2 + 4 = 6 -> Zyklus 24
    inst = mini()
    ev, cycle = P.fixed_line_times(inst.lines[0], 12, 2, 2, 0, 0, 0)
    assert ev[("dep", "b", 2)] == 10 and ev[("arr", "b", 1)] == 13 and ev[("dep", "b", 1)] == 15 and ev[("arr", "b", 0)] == 18 and cycle == 24
    ev4, _ = P.fixed_line_times(inst.lines[0], 12, 2, 2, 0, 4, 0)
    assert ev4[("dep", "b", 2)] == 14 and ev4[("arr", "b", 0)] == 22          # alle vier Minuten Schlupf am ersten Endpunkt


def test_lower_bound_equals_the_sum_of_pair_minima_over_all_offsets():
    for inst in (mini(), shared(), real_net(0, 3), real_net(1, 5)):
        total = 0.0
        for k1, k2, w1, w2 in P.reciprocal_pairs(inst):
            total += w1 * min(sum(pair_offsets_sum(inst, k1, k2, ob)) for ob in range(inst.T))
        assert abs(P.symmetric_lower_bound(inst) - total) < 1e-9


def test_greedy_places_the_heaviest_line_first_at_offset_zero():
    for seed in range(8):
        inst = real_net(seed)
        weight = np.zeros(len(inst.lines))
        for (a, _, _, b, _, _), w in inst.transfers.items():
            weight[a] += w
            weight[b] += w
        offsets = P.greedy_sequential(inst, P.base_arrays(inst))
        assert offsets[int(np.argmax(weight))] == 0, seed


def test_valid_splits_are_exactly_the_hand_derived_options():
    inst = mini()           # Linie 0: Schlupf 4, T = 12
    assert P.valid_splits(inst.lines[0], inst, False) == [(s, 0) for s in range(0, 5)]
    assert P.valid_splits(inst.lines[0], inst, True) == [(s, 0) for s in range(0, 5)] + [(s, 1) for s in range(5, 12)]


def test_candidate_matrix_with_headway_between_the_other_two_lines_at_the_boundary():
    """Drei Linien auf demselben Abschnitt: für die dritte Linie zählt das Paar der beiden anderen nur als Konstante - auch genau auf der Grenze d == hw."""
    inst = shared()
    inst = P.Instance(T=12, lines=inst.lines + [P.Line([4, 1, 2], [3, 4])], transfers=inst.transfers, dwell=2, tmin=2, transfer_min=3)
    cache = P.split_cache(inst, [False] * 3)
    pairs = P.shared_segments(inst)
    hw = 4
    boundary_seen = False
    for ob in range(12):
        o, si = [0, ob, 5], [0, 0, 0]
        cost, const = P.line_cost_matrix(inst, cache, o, si, 2, hw, pairs)
        for oc in range(12):
            o2 = [0, ob, oc]
            ref = P.cost_split(inst, cache, o2, si) + P.PEN * P.violations_split(inst, cache, o2, si, hw, pairs)
            assert abs(cost[oc, 0] + const - ref) < 1e-6, (ob, oc)
        d = (P.fixed_line_times(inst.lines[1], 12, 2, 2, ob)[0][("dep", "f", 1)] - P.fixed_line_times(inst.lines[0], 12, 2, 2, 0)[0][("dep", "f", 1)]) % 12
        boundary_seen |= min(d, 12 - d) == hw
    assert boundary_seen, "kein Versatz traf genau die Grenze: Test prüft die Grenze nicht"


def test_curve_step_zero_is_the_descent_of_the_given_start_not_a_new_search():
    inst = real_net(3)
    n = len(inst.lines)
    tw = sum(inst.transfers.values())
    cache = split_cache(inst, [False] * n)
    std = standard_split_indices(inst, cache)
    o0 = [0] * n
    expected = descent_split(inst, cache, o0, std)[2] / tw
    curve = budget_curve(inst, 1, 5, SplitMix64(9), init=[(o0, std)])
    assert abs(curve[0]["avg"] - expected) < 1e-9


def test_exact_fleet_cap_on_real_three_line_nets_matches_the_local_search():
    for seed, lower in ((100, 17.0), (103, 10.0)):
        inst = real_net(seed, 3)
        tw = sum(inst.transfers.values())
        n = len(inst.lines)
        cap0 = solve_pesp(inst, flex=False, flex_turn=True, fleet_extra=0, time_limit=30)
        cap1 = solve_pesp(inst, flex=False, flex_turn=True, fleet_extra=1, time_limit=30)
        assert cap0["status"] == cap1["status"] == "OPTIMAL"
        assert sum(cap0["fleet"]) == P.base_fleet(inst) and sum(cap1["fleet"]) > sum(cap0["fleet"])        # Deckel 0: Mindestbestand; Deckel 1 nutzt Zusatzzüge
        assert cap1["avg"] < cap0["avg"] - 1.0 and cap1["avg"] > lower
        _, _, c0, _ = local_search_split(inst, 40, random.Random(seed), [False] * n)
        _, _, c1, _ = local_search_split(inst, 40, random.Random(seed), [True] * n)
        assert abs(c0 / tw - cap0["avg"]) < 1e-6 and abs(c1 / tw - cap1["avg"]) < 1e-6                    # Lokalsuche = bewiesenes Optimum


def test_demand_greedy_without_demand_builds_no_line():
    coords = np.array([[10, 10], [20, 20], [30, 10]], dtype=np.int64)
    assert demand_greedy(coords, np.zeros((3, 3), dtype=np.int64), 3, 4) == [[], [], []]


def test_running_times_of_a_frozen_network_pin_the_rounding():
    inst = real_net(0)
    assert inst.lines[0].run == [12, 3, 7, 8, 5] and inst.lines[1].run == [2, 10, 4, 6, 7]


def test_default_shift_is_the_standard_split_and_changes_nothing():
    inst, _, _, _ = build_valid_instance(5, 11)
    for pair in sorted_pairs(inst)[:5]:
        lo, hi, std = pair_shift_range(inst, pair)
        assert lo == 0 and hi >= std == hi // 2
        assert pair_curves(inst, pair, (std, 0)) == pair_curves(inst, pair, None)
