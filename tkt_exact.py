"""Exakte Referenz: das volle PESP-Modell in OR-Tools CP-SAT (Phasen, Wende-Spielraum, optional Haltezeit-Spielraum,
Zugfolge, Fahrzeugdeckel). Liefert Fund, Schranke und Status; beweist beim erweiterten Modell selten."""
from __future__ import annotations

import math
import os
from collections import defaultdict

from ortools.sat.python import cp_model

from tkt_timetable import Instance, cycle_min, fixed_line_times, shared_segments, turn_slack

def solve_pesp(inst: Instance, flex: bool, headway: int = 0, fleet_extra: int | None = None,
               time_limit: float = 20.0, hint_offsets: list | None = None,
               flex_dwell: bool | None = None, flex_turn: bool | None = None, turn_extra_max: int | None = None) -> dict:
    """flex=False: Haltezeit und Wende fest (nur Linien-Offsets frei, Optimum der Offset-Aufgabe);
    flex=True: Halt in dwell_flex, Wende mit Spielraum (volles PESP). headway>0: Zugfolge auf geteilten Abschnitten.
    fleet_extra=None: Fahrzeuge frei; 0: höchstens das Minimum je Linie; k: Minimum + k."""
    T = inst.T
    fd = flex if flex_dwell is None else flex_dwell
    ft = flex if flex_turn is None else flex_turn
    t_extra = (T - 1) if turn_extra_max is None else turn_extra_max
    if headway > T // 2:
        raise ValueError("Zugfolge h muss <= T/2 sein (sonst leere Domäne)")
    model = cp_model.CpModel()
    pi = {}

    def ev(li, key):
        if (li, key) not in pi:
            pi[(li, key)] = model.NewIntVar(0, T - 1, f"pi_{li}_{key[0]}{key[1]}{key[2]}")
        return pi[(li, key)]

    def activity(u, v, lo, hi, name):
        x = model.NewIntVar(lo, hi, name)
        pmin = math.ceil((lo - (T - 1)) / T)
        pmax = math.floor((hi + (T - 1)) / T)
        p = model.NewIntVar(pmin, pmax, name + "_p")
        model.Add(v - u + T * p == x)
        return x

    cycle_x = defaultdict(list)
    for li, line in enumerate(inst.lines):
        m = len(line.stops) - 1
        slack = turn_slack(line, T, inst.dwell, inst.tmin)
        s_end = slack // 2
        dlo, dhi = inst.dwell_flex if fd else (inst.dwell, inst.dwell)
        for k in range(1, m + 1):
            cycle_x[li].append(activity(ev(li, ("dep", "f", k - 1)), ev(li, ("arr", "f", k)), line.run[k - 1], line.run[k - 1], f"run_f_{li}_{k}"))
            if k < m:
                cycle_x[li].append(activity(ev(li, ("arr", "f", k)), ev(li, ("dep", "f", k)), dlo, dhi, f"dw_f_{li}_{k}"))
        if ft:
            t_lo, t_hi = inst.tmin, inst.tmin + t_extra
        else:
            t_lo = t_hi = inst.tmin + s_end
        cycle_x[li].append(activity(ev(li, ("arr", "f", m)), ev(li, ("dep", "b", m)), t_lo, t_hi, f"turn_end_{li}"))
        for k in range(m - 1, -1, -1):
            cycle_x[li].append(activity(ev(li, ("dep", "b", k + 1)), ev(li, ("arr", "b", k)), line.run[k], line.run[k], f"run_b_{li}_{k}"))
            if k > 0:
                cycle_x[li].append(activity(ev(li, ("arr", "b", k)), ev(li, ("dep", "b", k)), dlo, dhi, f"dw_b_{li}_{k}"))
        if ft:
            t_lo, t_hi = inst.tmin, inst.tmin + t_extra
        else:
            t_lo = t_hi = inst.tmin + slack - s_end
        cycle_x[li].append(activity(ev(li, ("arr", "b", 0)), ev(li, ("dep", "f", 0)), t_lo, t_hi, f"turn_start_{li}"))
        vmin = math.ceil(cycle_min(line, dlo, inst.tmin) / T)
        if fleet_extra is not None:
            model.Add(sum(cycle_x[li]) <= T * (vmin + fleet_extra))

    obj_terms = []
    tx = []
    for idx, ((la, da, ka, lb, db, kb), w) in enumerate(inst.transfers.items()):
        x = activity(ev(la, ("arr", da, ka)), ev(lb, ("dep", db, kb)), inst.transfer_min, inst.transfer_min + T - 1, f"tr_{idx}")
        tx.append((w, x))
        obj_terms.append(int(round(w * 100)) * x)

    if headway > 0:
        for (la, ea), (lb, eb) in shared_segments(inst):
            activity(ev(la, ea), ev(lb, eb), headway, T - headway, f"hw_{la}_{lb}_{ea}{eb}")

    model.Add(ev(0, ("dep", "f", 0)) == 0)
    model.Minimize(sum(obj_terms))
    if hint_offsets is not None and not flex:
        for li, o in enumerate(hint_offsets):
            evt, _ = fixed_line_times(inst.lines[li], T, inst.dwell, inst.tmin, o)
            for key, val in evt.items():
                model.AddHint(ev(li, key), val % T)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = max(1, min(8, os.cpu_count() or 1))
    solver.parameters.random_seed = 1
    status = solver.Solve(model)
    name = solver.StatusName(status)
    out = {"status": name, "time": solver.WallTime()}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tw = sum(w for w, _ in tx)
        waits = [(w, solver.Value(x)) for w, x in tx]
        cost = sum(w * x for w, x in waits)
        out.update({"cost": cost, "bound": solver.BestObjectiveBound() / 100.0, "avg": cost / tw if tw else 0.0, "waits": waits, "weight": tw})
        out["fleet"] = [sum(solver.Value(x) for x in cycle_x[li]) // T for li in range(len(inst.lines))]
        out["pi"] = {k: solver.Value(v) for k, v in pi.items()}
    return out
