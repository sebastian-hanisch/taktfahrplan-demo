"""Reproduktion der Messreihe (Minuten bis Stunden, nicht in der CI). Teilaufgaben als Unterbefehle, jede schreibt eine JSONL-Datei
(Pfad = letzter Parameter); `tools/dump_sweep.py` fasst alles zu `data/tkt_results.json` zusammen.

  python tools/sweep.py nets  <Linien> <Ausgabe.jsonl>      30 Netze je Linienzahl (Seeds ab SWEEP_SEED_START), alle Stufen, Zugfolge 3/4/5
  python tools/sweep.py cp    <Ausgabe.jsonl> [Netze]       CP-SAT (20 s) gegen die Lokalsuche, 6 Linien, die ersten Netze der Hauptreihe
  python tools/sweep.py dwell <Ausgabe.jsonl> [Netze]       Haltezeit-Spielraum 1-3 min, nur CP-SAT (20 s), 6 Linien
  python tools/sweep.py noise <Ausgabe.jsonl>               5 Läufe je Netz, Spannweite von Aufteilung und Zusatzzug (6 Linien, 30 Netze)
  python tools/sweep.py recip <Ausgabe.jsonl>               Hin/Rück-Summe per Vollaufzählung des Versatzes an echten Paaren (6 Linien, 10 Netze)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkt_constants as C  # noqa: E402
from tkt_exact import solve_pesp  # noqa: E402
from tkt_network import build_instance_int  # noqa: E402
from tkt_rng import SplitMix64  # noqa: E402
from tkt_search import (PEN, base_arrays, budget_curve, descent_split_fixed_split, fleet_chosen, greedy_sequential,  # noqa: E402
                        local_search_multistart, local_search_split, random_offsets, split_cache, splits_of,
                        standard_split_indices, violations_split)
from tkt_timetable import (base_fleet, cycle_rank, evaluate_offsets, fixed_line_times, mean_wait, reciprocal_pairs, share_within,  # noqa: E402
                           shared_segments, symmetric_lower_bound, times_for, waits_for)

HW_STAGES = (3, 4, 5)


def summarize(inst, offsets, splits, tw):
    times = times_for(inst, offsets, splits)
    w = waits_for(inst, times)
    return {"avg": mean_wait(w), "le10": share_within(w, 10)}


def measure_net(n_lines, seed):
    inst, demand, _ = build_instance_int(C.N_STOPS, n_lines, C.MAX_LINE_LENGTH, seed)
    if not inst.transfers:
        return {"seed": seed, "skip": True}
    n = len(inst.lines)
    tw = sum(inst.transfers.values())
    arrs = base_arrays(inst)
    pairs = shared_segments(inst)
    rank, links = cycle_rank(inst)
    rng = SplitMix64(1_000_003 * seed + n_lines)
    rec = {"seed": seed, "n_lines": n, "links": links, "rank": rank, "base_fleet": base_fleet(inst), "n_recip_pairs": len(reciprocal_pairs(inst)),
           "n_segment_pairs": len(pairs), "transfer_share": tw / 2 / max(1.0, float(np.triu(demand, 1).sum()))}
    t0 = time.perf_counter()
    rnd = [evaluate_offsets(inst, random_offsets(n, inst.T, rng)) for _ in range(C.RANDOM_SAMPLES)]
    rec["random"] = {"avg": float(np.mean([r["avg"] for r in rnd])), "le10": float(np.mean([share_within(r["waits"], 10) for r in rnd]))}
    rec["same"] = summarize(inst, [0] * n, None, tw)
    rec["greedy"] = summarize(inst, greedy_sequential(inst, arrs), None, tw)
    o_ph, _ = local_search_multistart(inst, arrs, C.PHASE_RESTARTS, rng)
    rec["phase"] = summarize(inst, o_ph, None, tw)
    rec["lower_bound"] = symmetric_lower_bound(inst) / tw
    o, si, c, cache = local_search_split(inst, C.SEARCH_RESTARTS, rng, [False] * n)
    rec["split"] = {**summarize(inst, o, splits_of(cache, si), tw), "fleet": fleet_chosen(inst, cache, si)}
    o2, si2, c2, cache2 = local_search_split(inst, C.SEARCH_RESTARTS, rng, [True] * n, init=[(o, si)])
    rec["add"] = {**summarize(inst, o2, splits_of(cache2, si2), tw), "fleet": fleet_chosen(inst, cache2, si2)}
    curve = budget_curve(inst, n, C.CURVE_RESTARTS_SWEEP, rng, init=[(o, si)])
    rec["curve"] = [{"avg": e["avg"], "fleet": e["fleet"]} for e in curve]
    rec["t_core"] = time.perf_counter() - t0
    # Zugfolge 3/4/5
    cache0 = split_cache(inst, [False] * n)
    std = standard_split_indices(inst, cache0)
    rec["hw"] = {}
    for h in HW_STAGES:
        t1 = time.perf_counter()
        oh, sh, ch, cacheh = local_search_split(inst, C.SEARCH_RESTARTS, rng, [False] * n, hw=h)
        d = {"split": {**summarize(inst, oh, splits_of(cacheh, sh), tw), "feasible": violations_split(inst, cacheh, oh, sh, h, pairs) == 0}}
        oh2, sh2, ch2, cacheh2 = local_search_split(inst, C.SEARCH_RESTARTS, rng, [True] * n, init=[(oh, sh)], hw=h)
        d["add"] = {**summarize(inst, oh2, splits_of(cacheh2, sh2), tw), "feasible": violations_split(inst, cacheh2, oh2, sh2, h, pairs) == 0}
        best = None
        for _ in range(C.PHASE_RESTARTS):
            o3, _, c3 = descent_split_fixed_split(inst, cache0, [rng.randrange(inst.T) for _ in range(n)], std, h, pairs)
            if best is None or c3 < best[1]:
                best = (o3, c3)
        d["phase"] = {"avg": best[1] / tw if best[1] < PEN else None, "feasible": best[1] < PEN}
        d["time"] = time.perf_counter() - t1
        rec["hw"][str(h)] = d
    # Verletzungen der Aufteilungs-Lösung ohne Bedingung
    times = times_for(inst, o, splits_of(cache, si))
    rec["viol_free"] = {}
    for h in HW_STAGES:
        bad = 0
        for (la, ea), (lb, eb) in pairs:
            d_ = (times[lb][eb] - times[la][ea]) % inst.T
            bad += min(d_, inst.T - d_) < h
        rec["viol_free"][str(h)] = bad
    return rec


def run_nets(n_lines, out):
    s0 = C.SWEEP_SEED_START[n_lines]
    with open(out, "a", encoding="utf-8") as f:
        for seed in range(s0, s0 + C.SWEEP_NETS):
            t = time.time()
            rec = measure_net(n_lines, seed)
            rec["wall"] = time.time() - t
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if "skip" in rec:
                print(n_lines, seed, "keine Umsteiger", flush=True)
            else:
                print(n_lines, seed, round(rec["wall"], 1), "phase %.1f split %.1f add %.1f | h4 split %.1f" % (rec["phase"]["avg"], rec["split"]["avg"], rec["add"]["avg"], rec["hw"]["4"]["split"]["avg"]), flush=True)


def run_cp(out, n_nets):
    s0 = C.SWEEP_SEED_START[6]
    with open(out, "a", encoding="utf-8") as f:
        for seed in range(s0, s0 + n_nets):
            inst, _, _ = build_instance_int(C.N_STOPS, 6, C.MAX_LINE_LENGTH, seed)
            n = len(inst.lines)
            tw = sum(inst.transfers.values())
            rng = SplitMix64(1_000_003 * seed + 6)
            rec = {"seed": seed}
            for name, kw in (("split", dict(flex=False, flex_turn=True, fleet_extra=0)), ("phase", dict(flex=False))):
                r = solve_pesp(inst, time_limit=C.EXACT_TIME_LIMIT, **kw)
                rec[f"cp_{name}_status"] = r["status"]
                if "cost" in r:
                    rec[f"cp_{name}_avg"], rec[f"cp_{name}_bound"] = r["avg"], r["bound"] / tw
            o, si, c, cache = local_search_split(inst, C.SEARCH_RESTARTS, SplitMix64(31_000_003 * seed + C.SEARCH_RESTARTS), [False] * n)
            rec["ls_split_40"] = c / tw
            _, _, c20, _ = local_search_split(inst, 20, SplitMix64(31_000_003 * seed + 20), [False] * n)
            rec["ls_split_20"] = c20 / tw
            arrs = base_arrays(inst)
            _, cp_ph = local_search_multistart(inst, arrs, C.PHASE_RESTARTS, SplitMix64(5 + seed))
            rec["ls_phase"] = cp_ph / tw
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(seed, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)


def run_dwell(out, n_nets):
    s0 = C.SWEEP_SEED_START[6]
    with open(out, "a", encoding="utf-8") as f:
        for seed in range(s0, s0 + n_nets):
            inst, _, _ = build_instance_int(C.N_STOPS, 6, C.MAX_LINE_LENGTH, seed)
            n = len(inst.lines)
            tw = sum(inst.transfers.values())
            rng = SplitMix64(7_000_003 * seed + 6)
            rec = {"seed": seed, "base_fleet": base_fleet(inst)}
            o, si, c, cache = local_search_split(inst, C.SEARCH_RESTARTS, rng, [False] * n)
            rec["ls_split"] = c / tw
            o2, si2, c2, cache2 = local_search_split(inst, C.SEARCH_RESTARTS, rng, [True] * n, init=[(o, si)])
            rec["ls_add"] = c2 / tw
            for name, kw in (("cp_split0", dict(flex=False, flex_turn=True, fleet_extra=0)), ("cp_dwell_split0", dict(flex=True, fleet_extra=0)),
                             ("cp_dwell_split1", dict(flex=True, fleet_extra=1))):
                r = solve_pesp(inst, time_limit=C.EXACT_TIME_LIMIT, **kw)
                rec[name + "_status"] = r["status"]
                if "cost" in r:
                    rec[name] = r["avg"]
                    rec[name + "_bound"] = r["bound"] / tw
                    rec[name + "_fleet"] = int(sum(r["fleet"]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(seed, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)


def run_noise(out):
    s0 = C.SWEEP_SEED_START[6]
    with open(out, "a", encoding="utf-8") as f:
        for seed in range(s0, s0 + C.SWEEP_NETS):
            inst, _, _ = build_instance_int(C.N_STOPS, 6, C.MAX_LINE_LENGTH, seed)
            tw = sum(inst.transfers.values())
            n = len(inst.lines)
            sp, ad = [], []
            for rs in range(C.NOISE_RUNS):
                rng = SplitMix64(9_000_011 * seed + rs)
                o, si, c, _ = local_search_split(inst, C.SEARCH_RESTARTS, rng, [False] * n)
                sp.append(c / tw)
                _, _, c2, _ = local_search_split(inst, C.SEARCH_RESTARTS, rng, [True] * n, init=[(o, si)])
                ad.append(c2 / tw)
            rec = {"seed": seed, "split_spread": max(sp) - min(sp), "add_spread": max(ad) - min(ad)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec, flush=True)


def run_recip(out):
    s0 = C.SWEEP_SEED_START[6]
    rec = {"pairs": 0, "instances": 10, "violations": 0, "not_attainable": 0, "two_values_wrong": 0, "low_offsets": 0, "total_offsets": 0,
           "c_sum": 0, "c_zero": 0}
    for seed in range(s0, s0 + 10):
        inst, _, _ = build_instance_int(C.N_STOPS, 6, C.MAX_LINE_LENGTH, seed)
        T, mt = inst.T, inst.transfer_min
        ev0 = [fixed_line_times(l, T, inst.dwell, inst.tmin, 0)[0] for l in inst.lines]
        for k1, k2, w1, w2 in reciprocal_pairs(inst):
            la, da, ka, lb, db, kb = k1
            la2, da2, ka2, lb2, db2, kb2 = k2
            d1 = ev0[lb][("dep", db, kb)] - ev0[la][("arr", da, ka)] - mt
            d2 = ev0[la][("dep", db2, kb2)] - ev0[lb][("arr", da2, ka2)] - mt
            sums = [((ob + d1) % T + mt) + ((-ob + d2) % T + mt) for ob in range(T)]
            c = (d1 + d2) % T
            rec["pairs"] += 1
            rec["violations"] += any(v < 2 * mt + c for v in sums)
            rec["not_attainable"] += min(sums) != 2 * mt + c
            n_low = sum(1 for v in sums if v == 2 * mt + c)
            rec["two_values_wrong"] += (not set(sums) <= {2 * mt + c, 2 * mt + c + T}) or n_low != c + 1
            rec["low_offsets"] += n_low
            rec["total_offsets"] += T
            rec["c_sum"] += c
            rec["c_zero"] += c == 0
    Path(out).write_text(json.dumps(rec), encoding="utf-8")
    print(rec)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "nets":
        run_nets(int(sys.argv[2]), sys.argv[3])
    elif cmd == "cp":
        run_cp(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif cmd == "dwell":
        run_dwell(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
    elif cmd == "noise":
        run_noise(sys.argv[2])
    elif cmd == "recip":
        run_recip(sys.argv[2])
    else:
        raise SystemExit(__doc__)
