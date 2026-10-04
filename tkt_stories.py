"""Abnahmekriterien der Presets: jedes Preset erzählt eine Geschichte, die an der Messreihe UND am gezeigten Netz überprüfbar ist.

Die Kriterien sind reine Funktionen einfacher Zahlen (`facts`), damit Tests sie mit künstlichen Werten einzeln an ihrer Schwelle
kippen können; `facts_for` baut die Zahlen aus einer Live-Rechnung und der Ergebnisdatei. `tools/preset_search.py` sucht damit
Seeds außerhalb der Messreihen-Seeds, bei denen alle Kriterien des Presets gelten.
"""
from __future__ import annotations

import tkt_results as R

CRITERIA = {
    "Standard": [
        ("split_over_noise", "Aufteilungs-Gewinn des Netzes über der Meldungsschwelle", lambda f: f["split_gain"] > f["thr_split"]),
        ("le10_grows", "Anteil Umsteiger ≤ 10 min wächst von Phasen zu Aufteilung um mindestens 5 Prozentpunkte", lambda f: f["le10_split"] - f["le10_phase"] >= 0.05),
        ("greedy_close", "Messreihe: Greedy höchstens 5 % über dem Phasen-Optimum", lambda f: f["greedy_gap_pct"] <= 5.0),
    ],
    "Dichtes Netz": [
        ("ten_lines", "das Netz hat wirklich 10 Linien", lambda f: f["n_lines"] == 10),
        ("density_not_the_problem", "Messreihe: Phasen-Optimum bei 10 Linien höchstens 1,5 min über dem bei 3 Linien", lambda f: f["phase10_minus_phase3"] <= 1.5),
        ("headway_grows_with_density", "Messreihe: Zugfolge-Mehrkosten bei 10 Linien größer als bei 3 Linien", lambda f: f["hw_cost_10"] > f["hw_cost_3"]),
    ],
    "Kleines Netz": [
        ("three_lines", "das Netz hat wirklich 3 Linien", lambda f: f["n_lines"] == 3),
        ("low_rank", "Kreisrang des Umsteigegraphen höchstens 2", lambda f: f["rank"] <= 2),
        ("coupling_dominates", "Messreihe: Untergrenze erzwingt mindestens 70 % des Aufschlags über die Mindestzeit", lambda f: f["coupling_share_3"] >= 0.70),
        ("extra_train_over_noise", "ein Zusatzzug bringt mehr als die Meldungsschwelle des Zusatzzugs", lambda f: f["add_gain"] > f["thr_add"]),
    ],
    "Enge Zugfolge": [
        ("headway_over_noise", "Mehrkosten der Zugfolge auf dem Netz über der Meldungsschwelle", lambda f: f["headway_cost"] > f["thr_split"]),
        ("optimized_stages_feasible", "alle optimierten Stufen halten die Zugfolge ein", lambda f: f["violations_optimized"] == 0),
        ("headway_significant", "Messreihe: Mehrkosten bei 8 Linien und 5 min über 2 Standardfehlern", lambda f: f["hw_cost_8_h5"] > 2 * f["hw_cost_8_h5_se"]),
    ],
    "Viele Zusatzzüge": [
        ("saturation", "Gewinn des letzten Schritts der Kurve höchstens 0,2 min", lambda f: f["last_step_gain"] <= 0.2),
        ("trains_used", "Zusatzzüge genutzt: mindestens einer, höchstens 4", lambda f: 1 <= f["extra_trains"] <= 4),
        ("extra_train_over_noise", "alle Zusatzzüge zusammen bringen mehr als die Meldungsschwelle", lambda f: f["add_gain"] > f["thr_add"]),
        ("sweep_saturates", "Messreihe: Gewinn des letzten Kurvenschritts bei 6 Linien höchstens 0,2 min", lambda f: f["sweep_last_step_gain"] <= 0.2),
    ],
}


def sweep_facts(res: dict) -> dict:
    """Zahlen der Messreihe, die in den Kriterien vorkommen."""
    phase = R.size_curves(res)["phase"]
    p8 = R.headway_rows(res)
    row = next(r for r in p8 if r["lines"] == 8 and r["h"] == 5)
    cost = {r["lines"]: r["cost"] for r in p8 if r["h"] == 4}
    curve6 = R.curve_means(res, 6)
    return {"greedy_gap_pct": R.mean_se([(r["greedy"]["avg"] / r["phase"]["avg"] - 1) * 100 for r in R.nets(res)])[0],
            "phase10_minus_phase3": phase[10] - phase[3], "hw_cost_10": cost[10], "hw_cost_3": cost[3],
            "coupling_share_3": R.coupling_share(res, 3), "hw_cost_8_h5": row["cost"], "hw_cost_8_h5_se": row["se"],
            "sweep_last_step_gain": curve6[-2][2] - curve6[-1][2]}


def facts_for(run: dict, res: dict, thresholds: dict, free_split: dict | None = None) -> dict:
    """Zahlen eines Live-Laufs (Netz) zusammen mit den Messreihen-Zahlen."""
    st = run["stages"]
    curve = run["curve"]
    free = run["free_split"] if free_split is None else free_split
    return {**sweep_facts(res), "thr_split": thresholds["split"], "thr_add": thresholds["add"], "n_lines": len(run["inst"].lines),
            "split_gain": st["phase"]["avg"] - st["split"]["avg"], "add_gain": st["split"]["avg"] - st["add"]["avg"],
            "le10_phase": st["phase"]["le10"], "le10_split": st["split"]["le10"],
            "headway_cost": (st["split"]["avg"] - free["avg"]) if free else 0.0,
            "violations_optimized": sum(st[k]["violations"] or 0 for k in ("phase", "split", "add")),
            "last_step_gain": curve[-2]["avg"] - curve[-1]["avg"] if len(curve) > 1 else 0.0,
            "extra_trains": st["add"]["fleet"] - run["base_fleet"], "rank": run["rank"]}


def check(name: str, facts: dict) -> list:
    """[(Kennung, Text, erfüllt)] aller Kriterien des Presets."""
    return [(cid, text, bool(fn(facts))) for cid, text, fn in CRITERIA[name]]
