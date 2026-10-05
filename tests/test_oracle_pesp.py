"""Orakel-Tests (anderer Rechenweg): eigene Zeitrechnung und Aufzählung statt der Funktionen der Demo.

* Ereigniszeiten und Zyklus gegen einen frisch aufgebauten Zeitstrahl, Umsteigezeit per Schleife über [m, m+T-1],
* alle zulässigen Wende-Aufteilungen per Aufzählung der Standzeiten je Ende,
* Umsteiger gegen eine eigene Routenaufzählung,
* CP-SAT gegen die vollständige Aufzählung des PESP (Offsets, Haltezeiten 1-3, Wendezeiten) auf Mini-Netzen mit T = 6,
* Hin/Rück-Summenregel gegen Aufzählung aller 60 Versätze auf einem echten Netz.
"""
import itertools
import random

import numpy as np

from pesp_api import P
from tkt_network import build_valid_instance


def brute_wait(arr, dep, mt, T):
    return next(x for x in range(mt, mt + T) if (arr + x - dep) % T == 0)


def timeline(line, dwell, offset, turn_end, turn_start):
    m = len(line.stops) - 1
    segs = []
    for k in range(1, m + 1):
        segs.append((("arr", "f", k), line.run[k - 1]))
        if k < m:
            segs.append((("dep", "f", k), dwell))
    segs.append((("dep", "b", m), turn_end))
    for k in range(m - 1, -1, -1):
        segs.append((("arr", "b", k), line.run[k]))
        if k > 0:
            segs.append((("dep", "b", k), dwell))
    t = offset
    ev = {("dep", "f", 0): t}
    for key, d in segs:
        t += d
        ev[key] = t
    return ev, t + turn_start - offset


def rand_line(rng, pool, nmin, nmax, rmax):
    n = rng.randint(nmin, nmax)
    return P.Line(rng.sample(pool, n), [rng.randint(1, rmax) for _ in range(n - 1)])


def rand_inst(rng, n_lines, T, mt, nmax=3, rmax=3, npool=5):
    while True:
        pool = list(range(npool))
        lines = [rand_line(rng, pool, 2, nmax, rmax) for _ in range(n_lines)]
        demand = np.zeros((npool, npool))
        for i in range(npool):
            for j in range(i + 1, npool):
                if rng.random() < 0.7:
                    demand[i][j] = demand[j][i] = rng.randint(1, 9)
        tr = P.derive_transfers(lines, demand)
        if tr:
            return P.Instance(T=T, lines=lines, transfers=tr, dwell=2, tmin=2, transfer_min=mt, dwell_flex=(1, 3))


def test_event_times_wait_time_and_splits_against_enumeration():
    rng = random.Random(1)
    for _ in range(80):
        T = rng.choice([6, 8, 12, 60])
        tmin, dwell = rng.choice([2, 4]), rng.choice([1, 2])
        line = rand_line(rng, list(range(8)), 2, 6, 9)
        base = 2 * sum(line.run) + 2 * dwell * (len(line.stops) - 2)
        slack = next(s for s in range(T) if (base + 2 * tmin + s) % T == 0)
        assert P.turn_slack(line, T, dwell, tmin) == slack
        ev, cyc = P.fixed_line_times(line, T, dwell, tmin, 3)
        ev2, cyc2 = timeline(line, dwell, 3, tmin + slack // 2, tmin + slack - slack // 2)
        assert (ev, cyc) == (ev2, cyc2)
        arr, dep, mt = rng.randint(0, 3 * T), rng.randint(0, 3 * T), rng.choice([1, 3])
        assert P.wait_time(arr, dep, mt, T) == brute_wait(arr, dep, mt, T)
        # alle (Wendezeit Ende, Wendezeit Start) in [tmin, tmin+T-1] mit Zyklus % T == 0
        enum = {(a, b) for a in range(tmin, tmin + T) for b in range(tmin, tmin + T) if (base + a + b) % T == 0}
        inst = P.Instance(T=T, lines=[line], transfers={}, dwell=dwell, tmin=tmin, transfer_min=3)
        got = set()
        for s, e in P.valid_splits(line, inst, True):
            ev, cyc = P.fixed_line_times(line, T, dwell, tmin, 0, s, e)
            m = len(line.stops) - 1
            assert cyc % T == 0
            got.add((ev[("dep", "b", m)] - ev[("arr", "f", m)], cyc - ev[("arr", "b", 0)]))
        assert got == enum


def indep_transfers(lines, demand):
    out, ties = {}, 0
    n = demand.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(demand[i][j])
            if d <= 0 or any(i in l.stops and j in l.stops for l in lines):
                continue
            cands = []
            for la, A in enumerate(lines):
                for lb, B in enumerate(lines):
                    if la == lb or i not in A.stops or j not in B.stops:
                        continue
                    for s in set(A.stops) & set(B.stops):
                        ai, asx, bs, bj = A.stops.index(i), A.stops.index(s), B.stops.index(s), B.stops.index(j)
                        cost = sum(A.run[min(ai, asx):max(ai, asx)]) + sum(B.run[min(bs, bj):max(bs, bj)])
                        cands.append((cost, la, lb, asx, ai, bs, bj))
            if not cands:
                continue
            cands.sort()
            if len([c for c in cands if c[0] == cands[0][0]]) > 1:
                ties += 1
                continue
            _, la, lb, asx, ai, bs, bj = cands[0]
            dA, dB = ("f" if asx > ai else "b"), ("f" if bj > bs else "b")
            flip = {"f": "b", "b": "f"}
            k1, k2 = (la, dA, asx, lb, dB, bs), (lb, flip[dB], bs, la, flip[dA], asx)
            out[k1] = out.get(k1, 0) + d / 2
            out[k2] = out.get(k2, 0) + d / 2
    return out, ties


def test_derive_transfers_against_route_enumeration():
    rng = random.Random(2)
    checked = 0
    for _ in range(120):
        npool = rng.randint(5, 9)
        lines = [rand_line(rng, list(range(npool)), 2, 5, 6) for _ in range(rng.randint(2, 5))]
        demand = np.zeros((npool, npool))
        for i in range(npool):
            for j in range(i + 1, npool):
                if rng.random() < 0.8:
                    demand[i][j] = demand[j][i] = rng.randint(1, 9)
        got = P.derive_transfers(lines, demand)
        ref, ties = indep_transfers(lines, demand)
        if ties == 0:
            assert set(got) == set(ref) and all(abs(got[k] - ref[k]) < 1e-9 for k in ref)
            checked += 1
    assert checked > 30


def enum_pesp(inst, flex_dwell, flex_turn, headway, fleet_extra):
    """Vollständige Aufzählung (Offset der Linie 0 = 0), Zyklus % T == 0, je Wende in [tmin, tmin+T-1]."""
    T, mt = inst.T, inst.transfer_min
    dlo = inst.dwell_flex[0] if flex_dwell else inst.dwell
    cfgs = []
    for li, line in enumerate(inst.lines):
        m = len(line.stops) - 1
        n_dw = 2 * (m - 1)
        dws = list(itertools.product(range(inst.dwell_flex[0], inst.dwell_flex[1] + 1), repeat=n_dw)) if flex_dwell else [(inst.dwell,) * n_dw]
        vmin = -(-(2 * sum(line.run) + 2 * dlo * (m - 1) + 2 * inst.tmin) // T)
        items = []
        for dw in dws:
            base = 2 * sum(line.run) + sum(dw)
            slack0 = (-(base + 2 * inst.tmin)) % T
            for te in (range(inst.tmin, inst.tmin + T) if flex_turn else [inst.tmin + slack0 // 2]):
                if flex_turn:
                    ts = next((c for c in range(inst.tmin, inst.tmin + T) if (base + te + c) % T == 0), None)
                else:
                    ts = inst.tmin + slack0 - slack0 // 2
                if ts is None:
                    continue
                ev, cyc = timeline_dw(line, dw, te, ts)
                if fleet_extra is not None and cyc > T * (vmin + fleet_extra):
                    continue
                for o in ([0] if li == 0 else range(T)):
                    items.append({k: v + o for k, v in ev.items()})
        cfgs.append(items)
    segs = P.shared_segments(inst)
    best = None
    for evs in itertools.product(*cfgs):
        if headway > 0 and any(min((evs[lb][eb] - evs[la][ea]) % T, (evs[la][ea] - evs[lb][eb]) % T) < headway for (la, ea), (lb, eb) in segs):
            continue
        cost = sum(w * brute_wait(evs[la][("arr", da, ka)], evs[lb][("dep", db, kb)], mt, T) for (la, da, ka, lb, db, kb), w in inst.transfers.items())
        if best is None or cost < best:
            best = cost
    return best


def timeline_dw(line, dw, te, ts):
    m = len(line.stops) - 1
    t, ii = 0, 0
    ev = {("dep", "f", 0): 0}
    for k in range(1, m + 1):
        t += line.run[k - 1]
        ev[("arr", "f", k)] = t
        if k < m:
            t += dw[ii]
            ii += 1
            ev[("dep", "f", k)] = t
    t += te
    ev[("dep", "b", m)] = t
    for k in range(m - 1, -1, -1):
        t += line.run[k]
        ev[("arr", "b", k)] = t
        if k > 0:
            t += dw[ii]
            ii += 1
            ev[("dep", "b", k)] = t
    return ev, t + ts


def test_cpsat_equals_full_pesp_enumeration():
    rng = random.Random(3)
    kinds = [dict(flex=False), dict(flex=False, flex_turn=True, fleet_extra=0), dict(flex=True, fleet_extra=0), dict(flex=True, fleet_extra=1)]
    for t in range(24):
        two = t % 3 != 0
        inst = rand_inst(rng, 2 if two else 3, 6, rng.choice([1, 2]), nmax=3 if two else 2)
        kw = kinds[t % 4]
        hw = rng.choice([0, 0, 2, 3])
        fd, ft = kw.get("flex", False), kw.get("flex_turn", kw.get("flex", False))
        ref = enum_pesp(inst, fd, ft, hw, kw.get("fleet_extra"))
        res = P.solve_pesp(inst, headway=hw, time_limit=30, **kw)
        if ref is None:
            assert res["status"] == "INFEASIBLE"
        else:
            assert res["status"] == "OPTIMAL" and abs(res["cost"] - ref) < 1e-6, (t, kw, hw, res.get("cost"), ref)


def test_reciprocal_sum_rule_on_real_network():
    inst, _, _, _ = build_valid_instance(6, 0)
    T, mt = inst.T, inst.transfer_min
    evs0 = [P.fixed_line_times(l, T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]
    pairs = P.reciprocal_pairs(inst)
    assert pairs
    for k1, k2, _, _ in pairs[:5]:
        la, da, ka, lb, db, kb = k1
        sums = []
        for ob in range(T):
            eb = {k: v + ob for k, v in evs0[lb].items()}
            x1 = brute_wait(evs0[la][("arr", da, ka)], eb[("dep", db, kb)], mt, T)
            x2 = brute_wait(eb[("arr", k2[1], k2[2])], evs0[la][("dep", k2[4], k2[5])], mt, T)
            sums.append(x1 + x2)
        lo = min(sums)
        assert set(sums) <= {lo, lo + T} and sums.count(lo) == lo - 2 * mt + 1
    rng = random.Random(4)
    bound = P.symmetric_lower_bound(inst)
    for _ in range(40):
        o = [rng.randrange(T) for _ in inst.lines]
        assert P.evaluate_offsets(inst, o)["cost"] >= bound - 1e-9
