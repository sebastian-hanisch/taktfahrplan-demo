"""Fahrplanmodell: Linien, Ereigniszeiten im Takt, Umsteiger, Wartezeiten, Hin/Rück-Kopplung (Untergrenze).

Jede Linie fährt in beide Richtungen im Takt T. Ereignisse = Ankunft/Abfahrt je Haltestelle und Richtung; Umstieg =
Aktivität Ankunft -> Abfahrt mit Spannung x = pi_j - pi_i + T*p in [m, m+T-1]. Kleine Einheiten mit expliziten Ein- und
Ausgaben (DEMO-PLAYBOOK 1b); kein Zufall, kein Zustand.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

T_DEFAULT = 60

@dataclass
class Line:
    stops: list          # Haltestellen-Indizes in Reihenfolge
    run: list            # Fahrzeit (Minuten) je Abschnitt k: stops[k] -> stops[k+1]


@dataclass
class Instance:
    T: int
    lines: list
    transfers: dict      # (la, da, ka, lb, db, kb) -> Gewicht; Ankunft (la,da,ka) -> Abfahrt (lb,db,kb)
    dwell: int = 2
    tmin: int = 4
    transfer_min: int = 3
    dwell_flex: tuple = (1, 3)


# ---------------------------------------------------------------- Einheiten: Zeitberechnung
def turn_slack(line: Line, T: int, dwell: int, tmin: int) -> int:
    """Wartezeit an den Endpunkten, die der Zyklus braucht, um ein Vielfaches von T zu werden."""
    m = len(line.stops) - 1
    c0 = 2 * sum(line.run) + 2 * dwell * (m - 1) + 2 * tmin
    return (-c0) % T


def fixed_line_times(line: Line, T: int, dwell: int, tmin: int, offset: int, split: int | None = None, extra: int = 0):
    """Absolute Minuten aller Ereignisse einer Linie bei fester Haltezeit; Rückgabe (dict, Zyklusdauer)."""
    m = len(line.stops) - 1
    slack = turn_slack(line, T, dwell, tmin) + T * extra
    s_end = slack // 2 if split is None else split
    turn_end, turn_start = tmin + s_end, tmin + slack - s_end
    ev = {}
    t = offset
    ev[("dep", "f", 0)] = t
    for k in range(1, m + 1):
        t += line.run[k - 1]
        ev[("arr", "f", k)] = t
        if k < m:
            t += dwell
            ev[("dep", "f", k)] = t
    t += turn_end
    ev[("dep", "b", m)] = t
    for k in range(m - 1, -1, -1):
        t += line.run[k]
        ev[("arr", "b", k)] = t
        if k > 0:
            t += dwell
            ev[("dep", "b", k)] = t
    cycle = t + turn_start - offset
    return ev, cycle


def cycle_min(line: Line, dwell_lo: int, tmin: int) -> int:
    m = len(line.stops) - 1
    return 2 * sum(line.run) + 2 * dwell_lo * (m - 1) + 2 * tmin


# ---------------------------------------------------------------- Einheiten: Umsteiger ableiten
def direction(pos_from: int, pos_to: int) -> str:
    return "f" if pos_to > pos_from else "b"


def opposite(d: str) -> str:
    return "b" if d == "f" else "f"


def run_between(line: Line, a: int, b: int) -> int:
    lo, hi = min(a, b), max(a, b)
    return sum(line.run[lo:hi])


def derive_transfers(lines: list, demand: np.ndarray) -> dict:
    """Je Haltestellenpaar ohne Direktlinie die Route mit kürzester Fahrzeit über genau einen Umstieg;
    Nachfrage je zur Hälfte in jede Richtung auf die Umstiegsaktivität Ankunft -> Abfahrt."""
    pos = [{s: k for k, s in enumerate(l.stops)} for l in lines]
    n = demand.shape[0]
    out = defaultdict(float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(demand[i][j])
            if d <= 0:
                continue
            li = [x for x in range(len(lines)) if i in pos[x]]
            lj = [x for x in range(len(lines)) if j in pos[x]]
            if set(li) & set(lj):
                continue
            best = None
            for la in li:
                for lb in lj:
                    for s in set(pos[la]) & set(pos[lb]):
                        cost = run_between(lines[la], pos[la][i], pos[la][s]) + run_between(lines[lb], pos[lb][s], pos[lb][j])
                        if best is None or cost < best[0]:
                            best = (cost, la, lb, s)
            if best is None:
                continue
            _, la, lb, s = best
            ai, as_, bs, bj = pos[la][i], pos[la][s], pos[lb][s], pos[lb][j]
            dA, dB = direction(ai, as_), direction(bs, bj)
            out[(la, dA, as_, lb, dB, bs)] += d / 2
            dB2, dA2 = opposite(dB), opposite(dA)
            out[(lb, dB2, bs, la, dA2, as_)] += d / 2
    return dict(out)


# ---------------------------------------------------------------- Einheiten: Bewertung fester Takte
def wait_time(arr_min: int, dep_min: int, mt: int, T: int) -> int:
    """Umsteigezeit inkl. Mindestzeit: kleinste Zahl >= mt, die zu (dep - arr) mod T passt."""
    return (dep_min - arr_min - mt) % T + mt


def evaluate_offsets(inst: Instance, offsets: list, splits: list | None = None) -> dict:
    """Gewichtete Umsteigezeit bei festen Haltezeiten und Linien-Offsets."""
    T = inst.T
    times = []
    for li, line in enumerate(inst.lines):
        ev, _ = fixed_line_times(line, T, inst.dwell, inst.tmin, offsets[li], None if splits is None else splits[li])
        times.append(ev)
    tot_w = tot = 0.0
    waits = []
    for (la, da, ka, lb, db, kb), w in inst.transfers.items():
        x = wait_time(times[la][("arr", da, ka)], times[lb][("dep", db, kb)], inst.transfer_min, T)
        tot += w * x
        tot_w += w
        waits.append((w, x))
    return {"cost": tot, "avg": tot / tot_w if tot_w else 0.0, "waits": waits, "weight": tot_w}


def share_within(waits: list, limit: int) -> float:
    tw = sum(w for w, _ in waits)
    return sum(w for w, x in waits if x <= limit) / tw if tw else 0.0

# ---------------------------------------------------------------- Einheiten: Zugfolge-Abschnitte
def shared_segments(inst: Instance) -> list:
    """Paare von Abfahrtsereignissen zweier Linien auf demselben gerichteten Abschnitt."""
    seg = defaultdict(list)
    for li, line in enumerate(inst.lines):
        m = len(line.stops) - 1
        for k in range(m):
            seg[(line.stops[k], line.stops[k + 1])].append((li, ("dep", "f", k)))
            seg[(line.stops[k + 1], line.stops[k])].append((li, ("dep", "b", k + 1)))
    pairs = []
    for key, lst in seg.items():
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                if lst[x][0] != lst[y][0]:
                    pairs.append((lst[x], lst[y]))
    return pairs


def base_fleet(inst: Instance) -> int:
    return sum(fixed_line_times(l, inst.T, inst.dwell, inst.tmin, 0)[1] // inst.T for l in inst.lines)


# ---------------------------------------------------------------- Hin/Rück-Kopplung (analytische Untergrenze)
def reciprocal_pairs(inst: Instance) -> list:
    """Paare gegenläufiger Umstiegsaktivitäten desselben Anschlusses (gleiches Gewicht je Richtung)."""
    out, seen = [], set()
    for key, w in inst.transfers.items():
        la, da, ka, lb, db, kb = key
        rev = (lb, opposite(db), kb, la, opposite(da), ka)
        if rev in inst.transfers and rev not in seen and key not in seen:
            seen.add(key)
            seen.add(rev)
            out.append((key, rev, w, inst.transfers[rev]))
    return out


def symmetric_lower_bound(inst: Instance) -> float:
    """Bei festen Haltezeiten und Wende-Aufteilung gilt je Hin/Rück-Paar: Summe beider Wartezeiten >= 2*mt + (D1+D2) mod T,
    unabhängig von allen Offsets. Rückgabe: gewichtete Summe dieser Untergrenzen (nur Paare mit gleichen Gewichten je Richtung)."""
    T, mt = inst.T, inst.transfer_min
    ev0 = [fixed_line_times(l, T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]

    def d_of(key):
        la, da, ka, lb, db, kb = key
        return ev0[lb][("dep", db, kb)] - ev0[la][("arr", da, ka)] - mt

    tot = 0.0
    for k1, k2, w1, w2 in reciprocal_pairs(inst):
        assert abs(w1 - w2) < 1e-9
        tot += w1 * (2 * mt + (d_of(k1) + d_of(k2)) % T)
    return tot


# ---------------------------------------------------------------- Einheiten: Bewertung mit Aufteilung und Zusatzzug
def times_for(inst: Instance, offsets: list, splits: list | None = None) -> list:
    """Ereigniszeiten je Linie. `splits[li] = (s_end, Zusatzzüge)` oder None (Standard: Schlupf halbiert, kein Zusatzzug)."""
    out = []
    for li, line in enumerate(inst.lines):
        s, e = (None, 0) if splits is None else splits[li]
        out.append(fixed_line_times(line, inst.T, inst.dwell, inst.tmin, offsets[li], s, e)[0])
    return out


def waits_for(inst: Instance, times: list) -> list:
    """(Gewicht, Umsteige-Wartezeit inkl. Mindestzeit) je Umstiegsaktivität bei gegebenen Ereigniszeiten."""
    return [(w, wait_time(times[la][("arr", da, ka)], times[lb][("dep", db, kb)], inst.transfer_min, inst.T))
            for (la, da, ka, lb, db, kb), w in inst.transfers.items()]


def mean_wait(waits: list) -> float:
    tw = sum(w for w, _ in waits)
    return sum(w * x for w, x in waits) / tw if tw else 0.0


def segment_violations(inst: Instance, times: list, hw: int, pairs: list | None = None) -> int:
    """Zahl der Abschnittspaare, deren Abfahrten (mod T) näher als hw beieinander liegen."""
    if hw <= 0:
        return 0
    pairs = shared_segments(inst) if pairs is None else pairs
    bad = 0
    for (la, ea), (lb, eb) in pairs:
        d = (times[lb][eb] - times[la][ea]) % inst.T
        bad += min(d, inst.T - d) < hw
    return bad


def pair_offsets_sum(inst: Instance, key1: tuple, key2: tuple, offset_b: int, split_b: tuple | None = None) -> tuple:
    """Hin/Rück-Paar (key1: la -> lb, key2: lb -> la): Wartezeiten beider Richtungen, wenn Linie la bei Phase 0 (Standard-Aufteilung)
    liegt und Linie lb bei `offset_b` mit Aufteilung `split_b` = (s_end, Zusatzzüge) (None: Standard). Rückgabe (x1, x2)."""
    la, da, ka, lb, db, kb = key1
    la2, da2, ka2, lb2, db2, kb2 = key2
    assert (la2, lb2) == (lb, la)
    ev_a = fixed_line_times(inst.lines[la], inst.T, inst.dwell, inst.tmin, 0)[0]
    s, e = (None, 0) if split_b is None else split_b
    ev_b = fixed_line_times(inst.lines[lb], inst.T, inst.dwell, inst.tmin, offset_b, s, e)[0]
    x1 = wait_time(ev_a[("arr", da, ka)], ev_b[("dep", db, kb)], inst.transfer_min, inst.T)
    x2 = wait_time(ev_b[("arr", da2, ka2)], ev_a[("dep", db2, kb2)], inst.transfer_min, inst.T)
    return x1, x2


def cycle_rank(inst):
    """Kreisrang des Umsteigegraphen (Linien als Knoten, Umsteigeverbindungen als Kanten): Kanten - Knoten + Komponenten."""
    parent = list(range(len(inst.lines)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = set()
    for (a, _, _, b, _, _), w in inst.transfers.items():
        if w > 0 and a != b:
            edges.add((min(a, b), max(a, b)))
    for a, b in edges:
        parent[find(a)] = find(b)
    comps = len({find(x) for x in range(len(inst.lines))})
    return len(edges) - len(inst.lines) + comps, len(edges)
