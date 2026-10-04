"""Vorgerechnete Messreihe (data/tkt_results.json, erzeugt mit tools/sweep.py + tools/dump_sweep.py): Laden und Auswerten.

Die App rechnet die Messreihe nie live; alle Statistiken (Mittel ± Standardfehler, Median, P10/P90, gepaarte Differenzen,
Meldungsschwelle) kommen aus den Einzelwerten je Netz in der Datei.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

import tkt_constants as C

ROOT = Path(__file__).resolve().parent
STAGES = ("same", "random", "greedy", "phase", "split", "add")


def load_results(path=None) -> dict:
    return json.loads(Path(path or ROOT / C.RESULTS_FILE).read_text(encoding="utf-8"))


def nets(res: dict, n_lines: int | None = None) -> list:
    """Netze einer Linienzahl; None = alle (gepoolt)."""
    if n_lines is None:
        return [r for k in res["nets"] for r in res["nets"][k]]
    return res["nets"][str(n_lines)]


def mean_se(xs) -> tuple:
    xs = list(xs)
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan"))


def describe(xs) -> dict:
    xs = list(xs)
    m, se = mean_se(xs)
    return {"n": len(xs), "mean": m, "se": se, "median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90))}


def stage_values(rows: list, stage: str) -> list:
    return [r[stage]["avg"] for r in rows]


def cascade(res: dict, n_lines: int | None = None) -> dict:
    """Je Stufe: Beschreibung der mittleren Wartezeit über die Netze, mittlerer Anteil ≤ 10 min, mittlerer Zugbestand."""
    rows = nets(res, n_lines)
    out = {}
    for s in STAGES:
        d = describe(stage_values(rows, s))
        d["le10"] = st.mean(r[s]["le10"] for r in rows)
        d["fleet"] = st.mean(r[s].get("fleet", r["base_fleet"]) for r in rows)
        out[s] = d
    out["lower_bound"] = describe([r["lower_bound"] for r in rows])
    return out


def paired(res: dict, n_lines: int | None = None, h: int = 4) -> dict:
    """Gepaarte Differenzen je Netz (Minuten): Gewinn der Aufteilung, der Zusatzzüge, Mehrkosten der Zugfolge h, Greedy-Aufschlag."""
    rows = nets(res, n_lines)
    return {
        "split_gain": describe([r["phase"]["avg"] - r["split"]["avg"] for r in rows]),
        "add_gain": describe([r["split"]["avg"] - r["add"]["avg"] for r in rows]),
        "headway_cost": describe([r["hw"][str(h)]["split"]["avg"] - r["split"]["avg"] for r in rows]),
        "greedy_gap": describe([r["greedy"]["avg"] - r["phase"]["avg"] for r in rows]),
        "positive_split": sum(1 for r in rows if r["phase"]["avg"] - r["split"]["avg"] > 1e-9),
        "n": len(rows),
        "min_split_gain": min(r["phase"]["avg"] - r["split"]["avg"] for r in rows),
    }


def size_curves(res: dict) -> dict:
    """Mittlere Wartezeit je Stufe über die Netzgröße: {Stufe: {Linien: Mittel}}."""
    out = {s: {} for s in ("phase", "split", "add")}
    for k in res["nets"]:
        for s in out:
            out[s][int(k)] = st.mean(stage_values(res["nets"][k], s))
    return out


def coupling_share(res: dict, n_lines: int | None = None) -> float:
    """Anteil des Aufschlags über die Mindestzeit (3 min), den die Hin/Rück-Kopplung allein erzwingt (Mittel über die Netze)."""
    rows = nets(res, n_lines)
    return st.mean((r["lower_bound"] - C.TRANSFER_MIN) / (r["phase"]["avg"] - C.TRANSFER_MIN) for r in rows)


def headway_rows(res: dict) -> list:
    """Je Linienzahl und Stufe h: Mehrkosten (Mittel ± SE), % darüber, zulässig gefunden, Anteil Netze / Abschnittspaare verletzt (ohne Bedingung)."""
    rows = []
    for k in sorted(res["nets"], key=int):
        rs = res["nets"][k]
        for h in (3, 4, 5):
            cost = [r["hw"][str(h)]["split"]["avg"] - r["split"]["avg"] for r in rs]
            pct = [(r["hw"][str(h)]["split"]["avg"] / r["split"]["avg"] - 1) * 100 for r in rs]
            m, se = mean_se(cost)
            shares = [r["viol_free"][str(h)] / r["n_segment_pairs"] * 100 for r in rs if r["n_segment_pairs"]]
            rows.append({"lines": int(k), "h": h, "cost": m, "se": se, "pct": st.mean(pct),
                         "feasible": sum(1 for r in rs if r["hw"][str(h)]["split"]["feasible"]), "n": len(rs),
                         "nets_violated": sum(1 for r in rs if r["viol_free"][str(h)] > 0),
                         "pair_share": st.mean(shares) if shares else 0.0})
    return rows


def curve_means(res: dict, n_lines: int) -> list:
    """Mittlere Kurve Wartezeit gegen Zusatzzug-Erlaubnis: [(Erlaubnis k, Ø Zusatzzüge, Ø Wartezeit, Netze)]."""
    rows = nets(res, n_lines)
    out = []
    k = 0
    while True:
        pts = [(r["curve"][k]["fleet"] - r["base_fleet"], r["curve"][k]["avg"]) for r in rows if len(r["curve"]) > k]
        if not pts:
            break
        out.append((k, st.mean(p[0] for p in pts), st.mean(p[1] for p in pts), len(pts)))
        k += 1
    return out


def noise_thresholds(res: dict) -> dict:
    """Meldungsschwelle = P95 der Spannweite zweier Läufe der Lokalsuche (Aufteilung, Zusatzzug)."""
    return {"split": res["noise"]["split"]["p95"], "add": res["noise"]["add"]["p95"]}


def cp_summary(res: dict) -> dict:
    cp = res["cp"]
    ls40 = [r["ls_split_40"] for r in cp]
    cps = [r["cp_split_avg"] for r in cp]
    return {"n": len(cp), "cp_mean": st.mean(cps), "bound_mean": st.mean(r["cp_split_bound"] for r in cp),
            "ls40_mean": st.mean(ls40), "ls20_mean": st.mean(r["ls_split_20"] for r in cp),
            "proven": sum(1 for r in cp if r["cp_split_status"] == "OPTIMAL"),
            "ls40_worse": sum(1 for a, b in zip(ls40, cps) if a > b + 1e-6), "ls40_better": sum(1 for a, b in zip(ls40, cps) if a < b - 1e-6),
            "ls40_max_dev": max(a - b for a, b in zip(ls40, cps)),
            "ls20_worse": sum(1 for r in cp if r["ls_split_20"] > r["cp_split_avg"] + 1e-6),
            "ls20_max_dev": max(r["ls_split_20"] - r["cp_split_avg"] for r in cp),
            "phase_proven": sum(1 for r in cp if r["cp_phase_status"] == "OPTIMAL"),
            "phase_equal": sum(1 for r in cp if abs(r["ls_phase"] - r["cp_phase_avg"]) < 1e-6)}


def dwell_summary(res: dict) -> dict:
    d = res["dwell"]
    f = lambda k: [r[k] for r in d if k in r]
    return {"n": len(d), "split0": st.mean(f("cp_split0")), "dwell0": st.mean(f("cp_dwell_split0")), "dwell1": st.mean(f("cp_dwell_split1")),
            "ls_add": st.mean(f("ls_add")), "proven": [sum(1 for r in d if r.get(k + "_status") == "OPTIMAL") for k in ("cp_split0", "cp_dwell_split0", "cp_dwell_split1")],
            "bound0": st.mean(f("cp_dwell_split0_bound")), "bound1": st.mean(f("cp_dwell_split1_bound"))}


def nearest_lines(n_lines: int) -> int:
    return min(C.SWEEP_LINES, key=lambda k: (abs(k - n_lines), k))


def add_stage_at(res: dict, n_lines: int, k: int) -> dict:
    """Wartezeit der Stufe „+ Zusatzzug“ bei Erlaubnis für k Linien (Schritt k der Kurve), über die Netze gleicher Größe."""
    rows = nets(res, n_lines)
    return describe([r["curve"][min(k, len(r["curve"]) - 1)]["avg"] for r in rows])
