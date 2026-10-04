"""Lokalsuche für den Takt: Greedy, Phasen-Abstieg, Aufteilung der Wende-Standzeit, Zusatzzug, Mindest-Zugfolge.

Die Kandidatenmatrix `line_cost_matrix` bewertet für eine Linie alle (Phase, Aufteilung)-Paare auf einmal (numpy,
ohne Zufall); Zugfolge ist eine harte Bedingung per Strafterm PEN. Zufall nur über einen übergebenen Generator mit
`randrange` (SplitMix64 aus tkt_rng).
"""
from __future__ import annotations

import numpy as np

from tkt_timetable import Instance, Line, base_fleet, fixed_line_times, shared_segments, turn_slack

def base_arrays(inst: Instance):
    """Vorrechnung für schnelle Offset-Bewertung: wait = ((o_lb - o_la + D) mod T) + mt."""
    la, lb, D, W = [], [], [], []
    ev0 = [fixed_line_times(l, inst.T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]
    for (a, da, ka, b, db, kb), w in inst.transfers.items():
        la.append(a); lb.append(b); W.append(w)
        D.append(ev0[b][("dep", db, kb)] - ev0[a][("arr", da, ka)] - inst.transfer_min)
    return np.array(la), np.array(lb), np.array(D), np.array(W)


def fast_cost(arrs, offsets, T, mt) -> float:
    la, lb, D, W = arrs
    o = np.asarray(offsets)
    return float((W * (((o[lb] - o[la] + D) % T) + mt)).sum())


# ---------------------------------------------------------------- Einheiten: Heuristiken
def random_offsets(n_lines: int, T: int, rng) -> list:
    return [rng.randrange(T) for _ in range(n_lines)]


def greedy_sequential(inst: Instance, arrs) -> list:
    """Linien nach Umsteigegewicht absteigend einplanen; jede bekommt den Offset, der die Umsteigezeit zu
    den schon eingeplanten Linien minimiert."""
    la, lb, D, W = arrs
    T, mt = inst.T, inst.transfer_min
    n = len(inst.lines)
    weight = np.zeros(n)
    for a, b, w in zip(la, lb, W):
        weight[a] += w; weight[b] += w
    order = list(np.argsort(-weight, kind="stable"))
    placed = {}
    for li in order:
        if not placed:
            placed[li] = 0
            continue
        best_o, best_c = 0, float("inf")
        for o in range(T):
            c = 0.0
            for a, b, d, w in zip(la, lb, D, W):
                if a == li and b in placed:
                    c += w * (((placed[b] - o + d) % T) + mt)
                elif b == li and a in placed:
                    c += w * (((o - placed[a] + d) % T) + mt)
            if c < best_c:
                best_c, best_o = c, o
        placed[li] = best_o
    return [placed[i] for i in range(n)]


def coordinate_descent(inst: Instance, arrs, start: list) -> tuple:
    """Linie für Linie auf den besten Offset setzen, bis sich nichts mehr verbessert."""
    T, mt = inst.T, inst.transfer_min
    o = list(start)
    best = fast_cost(arrs, o, T, mt)
    improved = True
    while improved:
        improved = False
        for li in range(len(o)):
            keep = o[li]
            for cand in range(T):
                o[li] = cand
                c = fast_cost(arrs, o, T, mt)
                if c < best - 1e-9:
                    best, keep, improved = c, cand, True
            o[li] = keep
    return o, best


def local_search_multistart(inst: Instance, arrs, restarts: int, rng) -> tuple:
    best_o, best_c = None, float("inf")
    for _ in range(restarts):
        o, c = coordinate_descent(inst, arrs, random_offsets(len(inst.lines), inst.T, rng))
        if c < best_c:
            best_o, best_c = o, c
    return best_o, best_c

# ---------------------------------------------------------------- Heuristik mit Wende-Aufteilung und Zusatzzügen
def valid_splits(line: Line, inst: Instance, may_add: bool) -> list:
    """Zulässige (s_end, Zusatzzüge e'): ohne Zusatzzug s in [0, Schlupf]; mit der Option 'höchstens ein Zusatzzug'
    zusätzlich s in (Schlupf, T-1] (Gesamtschlupf + T, je Ende höchstens T-1 Minuten Standzeit)."""
    slack = turn_slack(line, inst.T, inst.dwell, inst.tmin)
    out = [(s, 0) for s in range(0, min(slack, inst.T - 1) + 1)]
    if may_add:
        out += [(s, 1) for s in range(slack + 1, inst.T)]
    return out


def split_cache(inst: Instance, may_add: list | None = None) -> list:
    """cache[li] = (Liste (s, e'), Ereigniszeiten bei Offset 0 je Eintrag)."""
    may_add = may_add or [False] * len(inst.lines)
    out = []
    for li, line in enumerate(inst.lines):
        opts = valid_splits(line, inst, may_add[li])
        out.append((opts, [fixed_line_times(line, inst.T, inst.dwell, inst.tmin, 0, s, e)[0] for s, e in opts]))
    return out


def violations_split(inst: Instance, cache: list, offsets: list, sidx: list, hw: int, pairs: list) -> int:
    if hw <= 0:
        return 0
    T = inst.T
    bad = 0
    for (la, ea), (lb, eb) in pairs:
        d = ((offsets[lb] + cache[lb][1][sidx[lb]][eb]) - (offsets[la] + cache[la][1][sidx[la]][ea])) % T
        if min(d, T - d) < hw:
            bad += 1
    return bad


def cost_split(inst: Instance, cache: list, offsets: list, sidx: list) -> float:
    T, mt = inst.T, inst.transfer_min
    tot = 0.0
    for (la, da, ka, lb, db, kb), w in inst.transfers.items():
        a = offsets[la] + cache[la][1][sidx[la]][("arr", da, ka)]
        d = offsets[lb] + cache[lb][1][sidx[lb]][("dep", db, kb)]
        tot += w * (((d - a - mt) % T) + mt)
    return tot


PEN = 1e6


def line_cost_matrix(inst: Instance, cache: list, o: list, si: list, li: int, hw: int, pairs: list) -> tuple:
    """Kosten aller Kandidaten (Offset x Aufteilung) für Linie li bei festen anderen Linien.
    Rückgabe (Matrix [T, n_s], Konstante): Gesamtkosten = Matrix[o_c, s_c] + Konstante; die Konstante sind Umstiege und
    Zugfolge-Strafen, an denen li nicht beteiligt ist."""
    T, mt = inst.T, inst.transfer_min
    n_s = len(cache[li][0])
    oo = np.arange(T)[:, None]
    cost = np.zeros((T, n_s))
    const = 0.0
    for (la, da, ka, lb, db, kb), w in inst.transfers.items():
        if la == li and lb != li:
            other = o[lb] + cache[lb][1][si[lb]][("dep", db, kb)]
            base = np.array([cache[li][1][x][("arr", da, ka)] for x in range(n_s)])
            cost += w * (((other - (oo + base[None, :]) - mt) % T) + mt)
        elif lb == li and la != li:
            other = o[la] + cache[la][1][si[la]][("arr", da, ka)]
            base = np.array([cache[li][1][x][("dep", db, kb)] for x in range(n_s)])
            cost += w * ((((oo + base[None, :]) - other - mt) % T) + mt)
        else:
            a = o[la] + cache[la][1][si[la]][("arr", da, ka)]
            d = o[lb] + cache[lb][1][si[lb]][("dep", db, kb)]
            const += w * (((d - a - mt) % T) + mt)
    if hw > 0:
        for (la, ea), (lb, eb) in pairs:
            if la == li:
                t_own = oo + np.array([cache[li][1][x][ea] for x in range(n_s)])[None, :]
                d = (o[lb] + cache[lb][1][si[lb]][eb] - t_own) % T
                cost += PEN * (np.minimum(d, T - d) < hw)
            elif lb == li:
                t_own = oo + np.array([cache[li][1][x][eb] for x in range(n_s)])[None, :]
                d = (t_own - (o[la] + cache[la][1][si[la]][ea])) % T
                cost += PEN * (np.minimum(d, T - d) < hw)
            else:
                d = ((o[lb] + cache[lb][1][si[lb]][eb]) - (o[la] + cache[la][1][si[la]][ea])) % T
                const += PEN * (min(d, T - d) < hw)
    return cost, const


def descent_split(inst: Instance, cache: list, offsets: list, sidx: list, hw: int = 0, pairs: list | None = None) -> tuple:
    """Linie für Linie bester (Offset, Aufteilung) bei festen anderen Linien, bis keine Verbesserung.
    hw > 0: Zugfolge als harte Bedingung per Strafterm (PEN je verletztem Abschnittspaar)."""
    pairs = shared_segments(inst) if (hw > 0 and pairs is None) else (pairs or [])
    o, si = list(offsets), list(sidx)
    best = cost_split(inst, cache, o, si) + PEN * violations_split(inst, cache, o, si, hw, pairs)
    improved = True
    while improved:
        improved = False
        for li in range(len(inst.lines)):
            cost, _ = line_cost_matrix(inst, cache, o, si, li, hw, pairs)
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            old = (o[li], si[li])
            o[li], si[li] = int(idx[0]), int(idx[1])
            c = cost_split(inst, cache, o, si) + PEN * violations_split(inst, cache, o, si, hw, pairs)
            if c < best - 1e-9:
                best, improved = c, True
            else:
                o[li], si[li] = old
    return o, si, best


def local_search_split(inst: Instance, restarts: int, rng, may_add: list | None = None,
                       init: list | None = None, hw: int = 0) -> tuple:
    """Rückgabe (offsets, s-Indizes, Kosten inkl. Strafe, cache). init = [(offsets, s-Indizes), ...] als Warmstarts
    (Indizes müssen im Cache gültig sein; die 'ohne Zusatzzug'-Optionen stehen am Anfang der Liste)."""
    cache = split_cache(inst, may_add)
    pairs = shared_segments(inst) if hw > 0 else []
    best = (None, None, float("inf"), cache)
    starts = [(o0, s0) for o0, s0 in (init or [])]
    for _ in range(restarts):
        starts.append((random_offsets(len(inst.lines), inst.T, rng), [rng.randrange(len(c[0])) for c in cache]))
    for o0, s0 in starts:
        o, si, c = descent_split(inst, cache, o0, s0, hw, pairs)
        if c < best[2]:
            best = (o, si, c, cache)
    return best



def fleet_chosen(inst: Instance, cache: list, sidx: list) -> int:
    return base_fleet(inst) + sum(cache[li][0][sidx[li]][1] for li in range(len(inst.lines)))



def descent_split_fixed_split(inst: Instance, cache: list, offsets: list, sidx: list, hw: int, pairs: list) -> tuple:
    """Nur Phasen optimieren (Aufteilung je Linie fest auf sidx), optional mit Zugfolge."""
    fixed = [([cache[li][0][sidx[li]]], [cache[li][1][sidx[li]]]) for li in range(len(inst.lines))]
    o, _, c = descent_split(inst, fixed, offsets, [0] * len(inst.lines), hw, pairs)
    return o, sidx, c


# ---------------------------------------------------------------- Einheiten: Stufen der Kaskade
def standard_split_indices(inst: Instance, cache: list) -> list:
    """Index der Standard-Aufteilung (Schlupf halbiert, kein Zusatzzug) in der Optionsliste jeder Linie."""
    return [cache[li][0].index((turn_slack(inst.lines[li], inst.T, inst.dwell, inst.tmin) // 2, 0)) for li in range(len(inst.lines))]


def phase_search(inst: Instance, restarts: int, rng, hw: int = 0, pairs: list | None = None) -> tuple:
    """Nur Phasen optimieren, Aufteilung = Standard. Ohne Zugfolge der schnelle Abstieg über `fast_cost`, mit Zugfolge der
    Abstieg mit Strafterm. Rückgabe (Offsets, Kosten inkl. Strafe)."""
    if hw <= 0:
        return local_search_multistart(inst, base_arrays(inst), restarts, rng)
    cache0 = split_cache(inst, [False] * len(inst.lines))
    std = standard_split_indices(inst, cache0)
    best = None
    for _ in range(restarts):
        o, _, c = descent_split_fixed_split(inst, cache0, random_offsets(len(inst.lines), inst.T, rng), std, hw, pairs)
        if best is None or c < best[1]:
            best = (o, c)
    return best


def splits_of(cache: list, sidx: list) -> list:
    """Die gewählten (s_end, Zusatzzüge)-Paare je Linie."""
    return [cache[li][0][sidx[li]] for li in range(len(sidx))]


def budget_curve(inst: Instance, max_lines: int, restarts: int, rng, init: list | None = None, hw: int = 0) -> list:
    """Wartezeit gegen Zusatzzug-Erlaubnis: Schritt k erlaubt k Linien einen Zusatzzug (jeweils die Linie, die am meisten
    bringt), Warmstart aus dem vorigen Schritt, daher je Netz monoton nicht steigend. Schritt 0 = Aufteilung ohne Zusatzzug;
    mit `init` = [(Offsets, s-Indizes)] ist er genau diese Lösung (keine zusätzlichen Neustarts). Jeder Eintrag trägt
    seine Lösung: offsets, splits ((s_end, Zusatzzüge) je Linie), allowed_lines, tatsächlich genutzter Bestand `fleet`."""
    n = len(inst.lines)
    tw = sum(inst.transfers.values())
    pairs = shared_segments(inst) if hw > 0 else []
    allowed = [False] * n

    def entry(k, allowed_lines, o, si, c, cache):
        return {"allowed": k, "allowed_lines": list(allowed_lines), "fleet": fleet_chosen(inst, cache, si),
                "avg": cost_split(inst, cache, o, si) / tw, "violations": violations_split(inst, cache, o, si, hw, pairs),
                "feasible": c < PEN, "offsets": list(o), "splits": splits_of(cache, si)}

    o, si, c, cache = local_search_split(inst, 0 if init else restarts * 3, rng, allowed, init=init, hw=hw)
    curve = [entry(0, allowed, o, si, c, cache)]
    for k in range(1, max_lines + 1):
        best = None
        for li in range(n):
            if allowed[li]:
                continue
            trial = list(allowed)
            trial[li] = True
            o2, si2, c2, cache2 = local_search_split(inst, restarts, rng, trial, init=[(o, si)], hw=hw)
            if best is None or c2 < best[0]:
                best = (c2, trial, o2, si2, cache2)
        if best is None:
            break
        c, allowed, o, si, cache = best
        curve.append(entry(k, allowed, o, si, c, cache))
    return curve
