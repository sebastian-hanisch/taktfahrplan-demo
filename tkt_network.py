"""Ganzzahliger Netzgenerator: Haltestellen auf einem Raster, symmetrische ganzzahlige Nachfrage mit Hub-Konzentration, Linien
nach der Nachfrage-Greedy-Konstruktion (Port von `demand_greedy_construction` der transit-demo, gegen eingefrorene
Referenzlinien des Originals getestet), Fahrzeiten aus ganzzahligen Entfernungen (`isqrt`). Zufall nur über SplitMix64.
"""
from __future__ import annotations

from math import isqrt

import numpy as np

from tkt_rng import SplitMix64
from tkt_timetable import Instance, Line, derive_transfers


def generate_network(n_stops: int, seed: int, hub_pct: int = 50, n_hubs: int = 2):
    """Ganzzahlige Haltestellen (Raster 5..95) und symmetrische ganzzahlige Nachfrage; hub_pct = Hub-Konzentration in Prozent
    (50 entspricht hub_concentration 0,5 des Originals: Hub-Zeilen und -Spalten mit Faktor 1 + 6*0,5 = 4)."""
    rng = SplitMix64(seed)
    coords = np.array([[5 + rng.below(91), 5 + rng.below(91)] for _ in range(n_stops)], dtype=np.int64)
    base = [[0] * n_stops for _ in range(n_stops)]
    for i in range(n_stops):
        for j in range(i + 1, n_stops):
            base[i][j] = base[j][i] = 1 + rng.below(10)
    pool = list(range(n_stops))
    hubs = []
    for _ in range(max(1, min(n_hubs, n_stops))):
        hubs.append(pool.pop(rng.below(len(pool))))
    boost = 100 + 6 * hub_pct
    b = [boost if i in hubs else 100 for i in range(n_stops)]
    demand = np.zeros((n_stops, n_stops), dtype=np.int64)
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                demand[i, j] = (base[i][j] * b[i] * b[j] + 5000) // 10000
    return coords, demand, hubs


def sq_dist(coords, a: int, b: int) -> int:
    dx = int(coords[a][0]) - int(coords[b][0])
    dy = int(coords[a][1]) - int(coords[b][1])
    return dx * dx + dy * dy


def demand_greedy(coords, demand, n_lines: int, max_line_length: int) -> list:
    """Port von `demand_greedy_construction` (Wachstumsbonus 2,5 = 5/2; alle Bewertungen mit 2 multipliziert, damit ganzzahlig;
    Gleichstände wie im Original: der erste Kandidat in Index-Reihenfolge gewinnt; Endenwahl über Quadratdistanzen)."""
    n = len(coords)
    GROW, ONE = 5, 2
    lines, covered, network = [], set(), set()
    for line_num in range(n_lines):
        best_pair, best_score, best_pair_demand = None, -1, 0
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in covered:
                    continue
                i_in, j_in = i in network, j in network
                if line_num > 0 and not (i_in or j_in):
                    continue
                d = int(demand[i][j])
                score = ONE * d
                if line_num > 0 and (i_in != j_in):
                    score = GROW * d
                if score > best_score:
                    best_score, best_pair, best_pair_demand = score, (i, j), d
        if best_pair is None or best_pair_demand <= 0:
            break
        line = list(best_pair)
        while len(line) < max_line_length:
            best_ext, best_ext_score = None, 0
            for cand in range(n):
                if cand in line:
                    continue
                gain = sum(int(demand[cand][s]) for s in line)
                score = gain * (GROW if cand not in network else ONE)
                if score > best_ext_score:
                    best_ext_score, best_ext = score, cand
            if best_ext is None:
                break
            if sq_dist(coords, best_ext, line[0]) <= sq_dist(coords, best_ext, line[-1]):
                line.insert(0, best_ext)
            else:
                line.append(best_ext)
        lines.append(line)
        network.update(line)
        for a in range(len(line)):
            for c in range(a + 1, len(line)):
                covered.add(tuple(sorted((line[a], line[c]))))
    while len(lines) < n_lines:
        lines.append([])
    return lines


def build_instance_int(n_stops: int, n_lines: int, max_len: int, seed: int, T: int = 60, hub_pct: int = 50):
    """Netz + Linien + Umsteiger für den Takt; Fahrzeit = Entfernung (isqrt, ganzzahlig) / 5 gerundet, mindestens 2 min."""
    coords, demand, hubs = generate_network(n_stops, seed, hub_pct)
    raw = [l for l in demand_greedy(coords, demand, n_lines, max_len) if len(l) >= 2]
    lines = []
    for l in raw:
        run = [max(2, (isqrt(sq_dist(coords, a, b)) + 2) // 5) for a, b in zip(l, l[1:])]
        lines.append(Line([int(x) for x in l], run))
    return Instance(T, lines, derive_transfers(lines, demand)), demand, coords


def build_valid_instance(n_lines: int, seed: int, n_stops: int = 20, max_len: int = 6, min_transfers: int = 3, attempts: int = 50):
    """Der erste gültige Seed ab `seed`: gültig = mindestens `min_transfers` Umsteigeverbindungen (sonst gäbe es nichts zu takten).
    Rückgabe (Instance, Nachfrage, Koordinaten, tatsächlich verwendeter Seed); die Versuche zählen den Seed einfach hoch."""
    for k in range(attempts):
        inst, demand, coords = build_instance_int(n_stops, n_lines, max_len, seed + k)
        if len(inst.lines) >= 2 and len(inst.transfers) >= min_transfers:
            return inst, demand, coords, seed + k
    raise ValueError(f"kein gültiger Seed ab {seed} in {attempts} Versuchen")
