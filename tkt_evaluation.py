"""Kennzahlen und Live-Rechnung: alle Stufen der Kaskade für ein Netz, Meldungs-Verdikte, Kopplungs-Abschnitt.

Stufen: zur vollen Stunde | zufällige Phasen (Mittel) | Greedy | Phasen optimiert | + Aufteilung der Wende-Standzeit |
+ Zusatzzug (Erlaubnis für k Linien). Mit Mindest-Zugfolge h > 0 sind die optimierten Stufen an die Bedingung gebunden; die
Referenzstufen zeigen dann die Zahl verletzter Abschnittspaare.
"""
from __future__ import annotations

import time

import numpy as np

import tkt_constants as C
from tkt_exact import solve_pesp
from tkt_network import build_valid_instance
from tkt_rng import SplitMix64
from tkt_search import (base_arrays, budget_curve, fleet_chosen, greedy_sequential, local_search_split, phase_search,
                        random_offsets, splits_of)
from tkt_timetable import (base_fleet, cycle_rank, evaluate_offsets, fixed_line_times, mean_wait, pair_offsets_sum, reciprocal_pairs,
                           segment_violations, share_within, shared_segments, symmetric_lower_bound, times_for, turn_slack, waits_for)

STAGE_ORDER = ("same", "random", "greedy", "phase", "split", "add")
STAGE_LABELS = {"same": "zur vollen Stunde", "random": "zufällige Phasen", "greedy": "Greedy", "phase": "Phasen optimiert",
                "split": "+ Aufteilung Wende", "add": "+ Zusatzzug erlaubt"}
VIEW_TO_STAGE = {"Phasen optimiert": "phase", "+ Aufteilung": "split", "+ Zusatzzug": "add"}


def stage_from_schedule(inst, key, offsets, splits, hw, pairs, fleet=None):
    """Kennzahlen eines konkreten Fahrplans (Phasen + Aufteilungen)."""
    times = times_for(inst, offsets, splits)
    waits = waits_for(inst, times)
    return {"key": key, "label": STAGE_LABELS[key], "avg": mean_wait(waits), "le10": share_within(waits, 10),
            "fleet": base_fleet(inst) if fleet is None else fleet, "violations": segment_violations(inst, times, hw, pairs),
            "offsets": list(offsets), "splits": splits, "times": times, "waits": waits}


def run_live(n_lines: int, seed: int, headway: int, budget: int) -> dict:
    """Alle Stufen für das Netz des Seeds. Rückgabe: stages (dict nach Schlüssel), curve, lower_bound, Netz-Objekte, seconds."""
    t0 = time.perf_counter()
    inst, demand, coords, used_seed = build_valid_instance(n_lines, seed, C.N_STOPS, C.MAX_LINE_LENGTH, C.MIN_TRANSFERS, C.SEED_ATTEMPTS)
    n = len(inst.lines)
    tw = sum(inst.transfers.values())
    arrs = base_arrays(inst)
    pairs = shared_segments(inst)
    hw = headway
    rng = SplitMix64(1_000_003 * used_seed + n_lines)
    std_splits = None
    stages = {}

    stages["same"] = stage_from_schedule(inst, "same", [0] * n, std_splits, hw, pairs)
    rnd = [evaluate_offsets(inst, random_offsets(n, inst.T, rng)) for _ in range(C.RANDOM_SAMPLES)]
    stages["random"] = {"key": "random", "label": STAGE_LABELS["random"], "avg": float(np.mean([r["avg"] for r in rnd])),
                        "le10": float(np.mean([share_within(r["waits"], 10) for r in rnd])), "fleet": base_fleet(inst),
                        "violations": None, "offsets": None, "splits": None, "times": None, "waits": None}
    stages["greedy"] = stage_from_schedule(inst, "greedy", greedy_sequential(inst, arrs), std_splits, hw, pairs)
    o_ph, _ = phase_search(inst, C.PHASE_RESTARTS, rng, hw, pairs)
    stages["phase"] = stage_from_schedule(inst, "phase", o_ph, std_splits, hw, pairs)

    o, si, c, cache = local_search_split(inst, C.SEARCH_RESTARTS, rng, None, hw=hw)
    stages["split"] = stage_from_schedule(inst, "split", o, splits_of(cache, si), hw, pairs, fleet=fleet_chosen(inst, cache, si))
    stages["split"]["feasible"] = c < 1e6
    free = None
    if hw > 0:      # dieselbe Stufe ohne Zugfolge: für die Meldung "was kostet die Bedingung"
        of, sf, cf, cachef = local_search_split(inst, C.SEARCH_RESTARTS, rng, None, hw=0)
        free = stage_from_schedule(inst, "split", of, splits_of(cachef, sf), hw, pairs, fleet=fleet_chosen(inst, cachef, sf))
    init = [(o, si)]
    curve = budget_curve(inst, n, C.CURVE_RESTARTS_LIVE, rng, init=init, hw=hw)
    step = curve[min(budget, n)]
    stages["add"] = stage_from_schedule(inst, "add", step["offsets"], step["splits"], hw, pairs, fleet=step["fleet"])
    stages["add"]["allowed_lines"] = step["allowed_lines"]
    return {"inst": inst, "demand": demand, "coords": coords, "seed": used_seed, "stages": stages, "curve": curve, "free_split": free,
            "lower_bound": symmetric_lower_bound(inst) / tw, "n_pairs": len(reciprocal_pairs(inst)), "pairs_segments": pairs,
            "base_fleet": base_fleet(inst), "rank": cycle_rank(inst)[0], "transfer_weight": tw, "headway": hw, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------ Meldungen (drei Zustände an der Meldungsschwelle)
def verdict(gain: float, threshold: float, phrase_over: str, phrase_under: str):
    """Drei Zustände: negativ (Bedingung half nicht: gain <= 0), im Rauschen (0 < gain <= Schwelle), über der Schwelle."""
    if gain > threshold:
        return "over", phrase_over
    if gain > 0:
        return "noise", phrase_under
    return "none", phrase_under


def split_message(stages: dict, threshold: float):
    gain = stages["phase"]["avg"] - stages["split"]["avg"]
    return gain, verdict(
        gain, threshold,
        f"Auf diesem Netz bringt die freie Aufteilung der Wende-Standzeit {gain:.1f} min weniger Wartezeit als der beste Takt mit "
        f"Standard-Standzeit - bei gleicher Zugzahl.",
        f"Auf diesem Netz liegt die Aufteilung im Rauschen ({gain:.1f} min, unter der Meldungsschwelle von {threshold:.1f} min); "
        f"die Messreihe unten zeigt den Befund über viele Netze.")


def headway_message(stages: dict, free_split: dict | None, hw: int, threshold: float):
    """Was kostet die Mindest-Zugfolge? Nur mit hw > 0 und mit der freien Vergleichsstufe."""
    if hw <= 0 or free_split is None:
        return None
    cost = stages["split"]["avg"] - free_split["avg"]
    state, _ = verdict(cost, threshold, "", "")
    if state == "over":
        text = f"Die Mindest-Zugfolge von {hw} min kostet auf diesem Netz {cost:.1f} min Wartezeit gegenüber der Aufteilung ohne Bedingung."
    elif state == "noise":
        text = (f"Die Mindest-Zugfolge von {hw} min kostet hier nur {cost:.1f} min - unter der Meldungsschwelle von {threshold:.1f} min, "
                f"also im Rauschen der Lokalsuche.")
    else:
        text = f"Die Mindest-Zugfolge von {hw} min kostet auf diesem Netz nichts messbar (Unterschied {cost:.1f} min)."
    return cost, state, text


# ------------------------------------------------------------------ Kernabschnitt: Hin/Rück-Kopplung
def pair_label(inst, pair, coords=None) -> str:
    key, _, w, _ = pair
    la, da, ka, lb, db, kb = key
    return f"Haltestelle {inst.lines[la].stops[ka]}: Linie {la + 1} ↔ Linie {lb + 1} (Gewicht {w:.1f} je Richtung)"


def pair_curves(inst, pair, split_b=None):
    """Wartezeiten beider Richtungen über alle Versätze der zweiten Linie: (Hin, Rück, Summe, c, untere Summe, Zahl der kleinen Versätze)."""
    key1, key2, _, _ = pair
    T = inst.T
    hin, rueck = [], []
    for ob in range(T):
        x1, x2 = pair_offsets_sum(inst, key1, key2, ob, split_b)
        hin.append(x1)
        rueck.append(x2)
    sums = [a + b for a, b in zip(hin, rueck)]
    low = min(sums)
    return hin, rueck, sums, low - 2 * inst.transfer_min, low, sums.count(low)


def pair_shift_range(inst, pair):
    """Zulässige Aufteilungen (ohne Zusatzzug) der zweiten Linie des Paares: (min, max, Standard) der Standzeit am Endpunkt."""
    lb = pair[0][3]
    slack = turn_slack(inst.lines[lb], inst.T, inst.dwell, inst.tmin)
    return 0, slack, slack // 2


def sorted_pairs(inst) -> list:
    """Hin/Rück-Paare nach Gewicht absteigend (stabil nach Reihenfolge)."""
    return sorted(reciprocal_pairs(inst), key=lambda p: -p[2])


def view_schedule(inst, stage) -> list:
    """Abfahrtsminuten je Linie und Halt (Hinrichtung/Rückrichtung) für die Fahrplan-Tabelle: [(Linie, [(Halt, hin, rück), ...])]."""
    out = []
    for li, line in enumerate(inst.lines):
        ev = stage["times"][li]
        m = len(line.stops) - 1
        rows = []
        for k, s in enumerate(line.stops):
            hin = ev.get(("dep", "f", k))
            rueck = ev.get(("dep", "b", k))
            rows.append((s, None if hin is None else hin % inst.T, None if rueck is None else rueck % inst.T))
        out.append((li, rows))
    return out


def top_pairs_table(inst, stage, limit=8) -> list:
    """Die gewichtigsten Anschlüsse (Hin/Rück-Paare) mit Wartezeit je Richtung und Summe bei diesem Fahrplan, dazu die Untergrenze der Summe."""
    T, mt = inst.T, inst.transfer_min
    ev0 = [fixed_line_times(l, T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]
    rows = []
    for k1, k2, w, _ in sorted_pairs(inst)[:limit]:
        la, da, ka, lb, db, kb = k1
        la2, da2, ka2, lb2, db2, kb2 = k2
        t = stage["times"]
        x1 = (t[lb][("dep", db, kb)] - t[la][("arr", da, ka)] - mt) % T + mt
        x2 = (t[la][("dep", db2, kb2)] - t[lb][("arr", da2, ka2)] - mt) % T + mt
        d1 = ev0[lb][("dep", db, kb)] - ev0[la][("arr", da, ka)] - mt
        d2 = ev0[la][("dep", db2, kb2)] - ev0[lb][("arr", da2, ka2)] - mt
        rows.append({"Haltestelle": inst.lines[la].stops[ka], "Linien": f"{la + 1} <-> {lb + 1}", "Gewicht je Richtung": round(w, 1),
                     "Wartezeit hin (min)": x1, "Wartezeit zurück (min)": x2, "Summe (min)": x1 + x2,
                     "Untergrenze der Summe (Standard-Standzeit)": 2 * mt + (d1 + d2) % T})
    return rows


EXACT_KWARGS = {
    0: dict(flex=False),                                         # nur Phasen, Standard-Standzeit
    1: dict(flex=False, flex_turn=True, fleet_extra=0),          # + Aufteilung der Standzeit, gleiche Züge
    2: dict(flex=True, fleet_extra=0),                           # + Haltezeit 1-3 min, gleiche Züge
    3: dict(flex=True, fleet_extra=1),                           # + Haltezeit und je Linie ein Zusatzzug
}


def run_exact(n_lines: int, seed: int, headway: int, mode: int, time_limit: float) -> dict:
    """CP-SAT auf demselben Netz wie die Live-Rechnung. Rückgabe: Status, Fund (Ø Wartezeit), Schranke (Ø), Zugbestand, Zeit."""
    inst, _, _, used_seed = build_valid_instance(n_lines, seed, C.N_STOPS, C.MAX_LINE_LENGTH, C.MIN_TRANSFERS, C.SEED_ATTEMPTS)
    tw = sum(inst.transfers.values())
    r = solve_pesp(inst, headway=headway, time_limit=time_limit, **EXACT_KWARGS[mode])
    out = {"status": r["status"], "time": r["time"], "seed": used_seed, "mode": mode, "headway": headway}
    if "cost" in r:
        out.update({"avg": r["avg"], "bound": r["bound"] / tw, "fleet": int(sum(r["fleet"])), "base_fleet": base_fleet(inst)})
    return out
