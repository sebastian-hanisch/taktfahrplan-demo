"""Mini-Instanz-Tests (von Hand gerechnet) für Fahrplanmodell, Suche und CP-SAT-Referenz (DEMO-PLAYBOOK 1b)."""
import itertools
import random

import numpy as np

from pesp_api import P


def mini():
    lines = [P.Line([0, 1, 2], [3, 3]), P.Line([3, 1, 4], [2, 4]), P.Line([2, 4, 0], [3, 3])]
    demand = np.zeros((5, 5))
    for (a, b, d) in [(0, 3, 10), (2, 3, 6)]:
        demand[a][b] = demand[b][a] = d
    return P.Instance(T=12, lines=lines, transfers=P.derive_transfers(lines, demand), dwell=2, tmin=2, transfer_min=3, dwell_flex=(1, 3))


def test_turn_slack_and_cycle():
    inst = mini()
    # L0: 2*(3+3) + 2*2*(2-1) + 2*2 = 20 -> Schlupf (-20) mod 12 = 4, Zyklus 24 = 2 Fahrzeuge
    assert P.turn_slack(inst.lines[0], 12, 2, 2) == 4
    ev, cyc = P.fixed_line_times(inst.lines[0], 12, 2, 2, 0)
    assert cyc == 24 and cyc % 12 == 0
    # Vorwärts: dep0=0, arr1=3, dep1=5, arr2=8; Wende 2+2=4 -> dep_b2=12; arr_b1=15, dep_b1=17, arr_b0=20, +Wende 4 = 24
    assert ev[("dep", "f", 0)] == 0 and ev[("arr", "f", 1)] == 3 and ev[("dep", "f", 1)] == 5 and ev[("arr", "f", 2)] == 8
    assert ev[("dep", "b", 2)] == 12 and ev[("arr", "b", 1)] == 15 and ev[("dep", "b", 1)] == 17 and ev[("arr", "b", 0)] == 20


def test_wait_time():
    assert P.wait_time(8, 11, 3, 12) == 3      # genau Mindestzeit
    assert P.wait_time(8, 10, 3, 12) == 14     # knapp verpasst: fast ein ganzer Takt
    assert P.wait_time(8, 8, 3, 12) == 12      # gleiche Minute: 12 mod 12 + ... = (0-3)%12+3 = 12
    assert P.wait_time(0, 5, 3, 12) == 5


def test_transfers_hand_derived():
    inst = mini()
    tr = inst.transfers
    # 0 -> 3 über Haltestelle 1: L0 vorwärts bis Pos 1, L1 von Pos 1 (Halt 1) rückwärts zu Halt 3 (Pos 0)
    assert tr[(0, "f", 1, 1, "b", 1)] == 5.0
    # Rückrichtung 3 -> 0: L1 vorwärts bis Pos 1, L0 rückwärts ab Pos 1
    assert tr[(1, "f", 1, 0, "b", 1)] == 5.0
    # 2 -> 3 über Haltestelle 1 (Fahrzeit 5 < 9 über Halt 4): L0 rückwärts Pos 2 -> 1, L1 rückwärts Pos 1 -> 0
    assert tr[(0, "b", 1, 1, "b", 1)] == 3.0


def test_cpsat_equals_bruteforce_offset_model():
    inst = mini()
    T = inst.T
    best = min(P.evaluate_offsets(inst, list(o))["cost"] for o in itertools.product(range(T), repeat=3))
    res = P.solve_pesp(inst, flex=False, time_limit=20)
    assert res["status"] == "OPTIMAL"
    assert abs(res["cost"] - best) < 1e-6, (res["cost"], best)
    # unabhängige Kreuzprobe der Bewertung: CP-SAT-Wartezeiten sind die gleiche Formel wie evaluate_offsets
    arrs = P.base_arrays(inst)
    rng = random.Random(1)
    for _ in range(50):
        o = P.random_offsets(3, T, rng)
        assert abs(P.fast_cost(arrs, o, T, inst.transfer_min) - P.evaluate_offsets(inst, o)["cost"]) < 1e-6


def test_flex_never_worse_and_fleet_cap():
    inst = mini()
    fixed = P.solve_pesp(inst, flex=False, time_limit=20)
    flex = P.solve_pesp(inst, flex=True, time_limit=20)
    assert flex["status"] == "OPTIMAL" and flex["cost"] <= fixed["cost"] + 1e-6
    cap = P.solve_pesp(inst, flex=True, fleet_extra=0, time_limit=20)
    assert cap["status"] == "OPTIMAL" and cap["cost"] >= flex["cost"] - 1e-6
    # kleinstes Fahrzeugminimum: ceil(Zyklus_min/T) = ceil(20/12)=2 je Linie (Halt 1)
    assert all(v == 2 for v in cap["fleet"]), cap["fleet"]


def test_heuristics_feasible_and_not_below_optimum():
    inst = mini()
    arrs = P.base_arrays(inst)
    opt = P.solve_pesp(inst, flex=False, time_limit=20)["cost"]
    g = P.greedy_sequential(inst, arrs)
    ls, c_ls = P.local_search_multistart(inst, arrs, 10, random.Random(2))
    assert P.fast_cost(arrs, g, inst.T, inst.transfer_min) >= opt - 1e-6
    assert c_ls >= opt - 1e-6


def shared():
    # L0 und L1 teilen den Abschnitt 1-2 (beide Richtungen); Umsteiger 0 -> 3 über Halt 1 und 2 -> 3
    lines = [P.Line([0, 1, 2], [3, 3]), P.Line([3, 1, 2], [2, 4])]
    demand = np.zeros((4, 4))
    demand[0][3] = demand[3][0] = 10
    return P.Instance(T=12, lines=lines, transfers=P.derive_transfers(lines, demand), dwell=2, tmin=2, transfer_min=3)


def test_headway_enforced_and_costly():
    inst = shared()
    segs = P.shared_segments(inst)
    assert len(segs) == 2                      # (1,2) vorwärts und (2,1) rückwärts
    free = P.solve_pesp(inst, flex=True, headway=0, time_limit=20)
    hw = P.solve_pesp(inst, flex=True, headway=4, time_limit=20)
    assert free["status"] == "OPTIMAL" and hw["status"] == "OPTIMAL" and hw["cost"] >= free["cost"] - 1e-6
    for (la, ea), (lb, eb) in segs:
        diff = (hw["pi"][(lb, eb)] - hw["pi"][(la, ea)]) % inst.T
        assert 4 <= diff <= inst.T - 4, diff
    try:
        P.solve_pesp(inst, flex=True, headway=7, time_limit=20)
        raise AssertionError("h > T/2 muss abgelehnt werden")
    except ValueError:
        pass
    # drei Linien auf demselben Abschnitt: 3*4 = 12 = T passt gerade, 3*5 > T ist unlösbar
    lines = inst.lines + [P.Line([4, 1, 2], [3, 4])]
    three = P.Instance(T=12, lines=lines, transfers=inst.transfers, dwell=2, tmin=2, transfer_min=3)
    assert P.solve_pesp(three, flex=True, headway=4, time_limit=20)["status"] == "OPTIMAL"
    assert P.solve_pesp(three, flex=True, headway=5, time_limit=20)["status"] == "INFEASIBLE"


def test_split_heuristic_matches_exact_neighbourhood():
    inst = mini()
    T = inst.T
    cache = P.split_cache(inst)
    best = min(P.cost_split(inst, cache, list(o), list(sp))
               for o in itertools.product(range(T), repeat=3)
               for sp in itertools.product(*[range(len(c[0])) for c in cache]))
    exact = P.solve_pesp(inst, flex=False, flex_turn=True, fleet_extra=0, time_limit=30)
    assert exact["status"] == "OPTIMAL" and abs(exact["cost"] - best) < 1e-6, (exact["cost"], best)
    _, _, c, _ = P.local_search_split(inst, 30, random.Random(3))
    assert c >= best - 1e-6
    # Konsistenz: Standard-Aufteilung entspricht evaluate_offsets
    o = [1, 5, 7]
    std = []
    for li, (opts, _) in enumerate(cache):
        slack = P.turn_slack(inst.lines[li], T, inst.dwell, inst.tmin)
        std.append(opts.index((slack // 2, 0)))
    assert abs(P.cost_split(inst, cache, o, std) - P.evaluate_offsets(inst, o)["cost"]) < 1e-6


def test_may_add_vehicle_matches_cpsat_and_is_superset():
    # zwei Linien: Vollaufzählung (Offsets x Aufteilungen) gegen Heuristik und CP-SAT (Deckel Minimum+1)
    full = mini()
    inst = P.Instance(T=12, lines=full.lines[:2], transfers={k: v for k, v in full.transfers.items() if k[0] < 2 and k[3] < 2},
                      dwell=2, tmin=2, transfer_min=3)
    assert inst.transfers
    T = 12
    results = []
    for may in ([False, False], [True, False], [True, True]):
        cache = P.split_cache(inst, may)
        best = min(P.cost_split(inst, cache, list(o), list(sp))
                   for o in itertools.product(range(T), repeat=2)
                   for sp in itertools.product(*[range(len(c[0])) for c in cache]))
        _, _, c, _ = P.local_search_split(inst, 40, random.Random(4), may)
        assert abs(c - best) < 1e-6, (may, c, best)
        results.append(best)
    assert results[0] >= results[1] - 1e-6 >= results[2] - 2e-6     # mehr Erlaubnis = Obermenge, nie schlechter
    exact = P.solve_pesp(inst, flex=False, flex_turn=True, fleet_extra=1, time_limit=30)
    assert exact["status"] == "OPTIMAL" and abs(exact["cost"] - results[2]) < 1e-6, (exact["cost"], results)
    # der tatsächlich genutzte Bestand zählt Zusatzzüge nur, wo s über dem Schlupf liegt
    cache = P.split_cache(inst, [True, True])
    for si in itertools.product(*[range(len(c[0])) for c in cache]):
        used = P.fleet_chosen(inst, cache, list(si))
        assert used == P.base_fleet(inst) + sum(cache[i][0][si[i]][1] for i in range(2))


def test_symmetric_pair_sum_is_fixed_by_running_times():
    inst = mini()
    T, mt = inst.T, inst.transfer_min
    pairs = P.reciprocal_pairs(inst)
    assert pairs, "Mini-Instanz hat Hin/Rück-Paare"
    k1, k2, w1, w2 = pairs[0]
    la, da, ka, lb, db, kb = k1
    la2, da2, ka2, lb2, db2, kb2 = k2
    ev0 = [P.fixed_line_times(l, T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]
    D1 = ev0[lb][("dep", db, kb)] - ev0[la][("arr", da, ka)] - mt
    D2 = ev0[lb2][("dep", db2, kb2)] - ev0[la2][("arr", da2, ka2)] - mt
    c = (D1 + D2) % T
    sums = []
    for ob in range(T):
        # Offsets der beiden Linien: la bei 0, lb bei ob (k2 läuft in Gegenrichtung lb -> la)
        x1 = (ob + D1) % T + mt
        x2 = (0 - ob + D2) % T + mt
        sums.append(x1 + x2)
    assert min(sums) == 2 * mt + c and all(v >= 2 * mt + c for v in sums), (min(sums), 2 * mt + c)
    # und die analytische Gesamtuntergrenze liegt nie über dem Optimum der Offset-Aufgabe
    opt = P.solve_pesp(inst, flex=False, time_limit=20)["cost"]
    assert P.symmetric_lower_bound(inst) <= opt + 1e-6


def test_headway_aware_search_respects_constraint_or_reports():
    inst = shared()
    T = 12
    o, si, c, cache = P.local_search_split(inst, 30, random.Random(5), None, hw=4)
    pairs = P.shared_segments(inst)
    assert P.violations_split(inst, cache, o, si, 4, pairs) == 0 and c < P.PEN
    # warm start from a solution is never worse than the same search without it
    o0, s0, c0, _ = P.local_search_split(inst, 5, random.Random(6))
    _, _, c1, _ = P.local_search_split(inst, 0, random.Random(6), None, init=[(o0, s0)])
    assert c1 <= c0 + 1e-9


def test_budget_curve_monotone_and_fixed_split_descent():
    inst = mini()
    curve = P.budget_curve(inst, 3, 4, random.Random(7))
    avgs = [x["avg"] for x in curve]
    assert all(b <= a + 1e-9 for a, b in zip(avgs, avgs[1:])), avgs
    assert curve[0]["fleet"] == P.base_fleet(inst) and all(x["fleet"] >= curve[0]["fleet"] for x in curve)
    # Phasen-Abstieg mit fester Standard-Aufteilung = Offset-Optimum der Vollaufzählung
    cache0 = P.split_cache(inst, [False] * 3)
    std = []
    for li, (opts, _) in enumerate(cache0):
        slack = P.turn_slack(inst.lines[li], 12, inst.dwell, inst.tmin)
        std.append(opts.index((slack // 2, 0)))
    best = min(P.evaluate_offsets(inst, list(o))["cost"] for o in itertools.product(range(12), repeat=3))
    got = min(P.descent_split_fixed_split(inst, cache0, [random.Random(i).randrange(12) for _ in range(3)], std, 0, [])[2] for i in range(20))
    assert abs(got - best) < 1e-6, (got, best)


def test_candidate_matrix_equals_full_reevaluation():
    """Delta-Prüfung (DEMO-PLAYBOOK 1b): jeder Eintrag der Kandidatenmatrix = volle Neubewertung mit Konstante."""
    full = mini()
    for inst, hw in [(full, 0), (shared(), 4)]:
        T = inst.T
        n = len(inst.lines)
        cache = P.split_cache(inst, [True] * n)
        pairs = P.shared_segments(inst)
        rng = random.Random(11)
        for _ in range(6):
            o = [rng.randrange(T) for _ in range(n)]
            si = [rng.randrange(len(c[0])) for c in cache]
            for li in range(n):
                cost, const = P.line_cost_matrix(inst, cache, o, si, li, hw, pairs)
                for _ in range(15):
                    oc, sc = rng.randrange(T), rng.randrange(len(cache[li][0]))
                    o2, si2 = list(o), list(si)
                    o2[li], si2[li] = oc, sc
                    ref = P.cost_split(inst, cache, o2, si2) + P.PEN * P.violations_split(inst, cache, o2, si2, hw, pairs)
                    assert abs(cost[oc, sc] + const - ref) < 1e-6, (li, oc, sc, cost[oc, sc] + const, ref)
