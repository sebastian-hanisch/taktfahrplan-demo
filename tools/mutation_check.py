"""Fehler-Einbau-Test: baut einzelne Fehler in die Module ein und prüft, ob die Tests (ohne AppTests) sie finden.

Aufruf (im Projektordner): python tools/mutation_check.py [Teilstring des Dateinamens] [--jobs N] [--indices 5,8-12] [--with-app] [--dry-run]
Jeder Mutant ersetzt genau eine Stelle; Überlebende sind entweder gleichwertig (kein sichtbarer Unterschied) oder eine Lücke der Tests. Die Kopie liegt je Mutant in einem
temporären Ordner; PYTHONDONTWRITEBYTECODE=1, damit veralteter Bytecode keine Überlebenden vortäuscht; Quelltexte als LF. Ein Mutant kann in eine Endlosschleife laufen;
nach TIMEOUT Sekunden gilt er als gefunden. `--with-app` nimmt die AppTests hinzu, `--dry-run` prüft nur, ob jede Zeichenkette genau einmal vorkommt."""
import concurrent.futures
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
WITH_APP = "--with-app" in sys.argv
TIMEOUT = 600
TEST_ORDER = ["test_results.py", "test_presets.py", "test_network.py", "test_pesp_core.py", "test_pdf.py", "test_stories.py", "test_search_branches.py", "test_evaluation.py", "test_gaps.py", "test_claims.py"]

MUTANTS = [
    ('tkt_timetable.py', '    return (-c0) % T', '    return c0 % T'),
    ('tkt_timetable.py', 's_end = slack // 2 if split is None else split', 's_end = slack // 3 if split is None else split'),
    ('tkt_timetable.py', 'turn_end, turn_start = tmin + s_end, tmin + slack - s_end', 'turn_end, turn_start = tmin + slack - s_end, tmin + s_end'),
    ('tkt_timetable.py', 'return (dep_min - arr_min - mt) % T + mt', 'return (dep_min - arr_min) % T + mt'),
    ('tkt_timetable.py', 'out[(la, dA, as_, lb, dB, bs)] += d / 2', 'out[(la, dA, as_, lb, dB, bs)] += d'),
    ('tkt_timetable.py', 'out[(lb, dB2, bs, la, dA2, as_)] += d / 2', 'out[(lb, dB2, bs, la, dA2, as_)] += d / 3'),
    ('tkt_timetable.py', '            if set(li) & set(lj):', '            if False:'),
    ('tkt_timetable.py', 'tot += w1 * (2 * mt + (d_of(k1) + d_of(k2)) % T)', 'tot += w1 * (2 * mt + d_of(k1) % T)'),
    ('tkt_timetable.py', 'bad += min(d, inst.T - d) < hw', 'bad += min(d, inst.T - d) <= hw'),
    ('tkt_timetable.py', '    return sum(w * x for w, x in waits) / tw if tw else 0.0', '    return sum(x for w, x in waits) / len(waits) if tw else 0.0'),
    ('tkt_timetable.py', 'x2 = wait_time(ev_b[("arr", da2, ka2)], ev_a[("dep", db2, kb2)], inst.transfer_min, inst.T)', 'x2 = wait_time(ev_a[("dep", db2, kb2)], ev_b[("arr", da2, ka2)], inst.transfer_min, inst.T)'),
    ('tkt_search.py', 'order = list(np.argsort(-weight, kind="stable"))', 'order = list(np.argsort(weight, kind="stable"))'),
    ('tkt_search.py', 'out += [(s, 1) for s in range(slack + 1, inst.T)]', 'out += [(s, 1) for s in range(slack, inst.T)]'),
    ('tkt_search.py', 'out = [(s, 0) for s in range(0, min(slack, inst.T - 1) + 1)]', 'out = [(s, 0) for s in range(0, min(slack, inst.T - 1))]'),
    ('tkt_search.py', 'return base_fleet(inst) + sum(cache[li][0][sidx[li]][1] for li in range(len(inst.lines)))', 'return base_fleet(inst) - sum(cache[li][0][sidx[li]][1] for li in range(len(inst.lines)))'),
    ('tkt_search.py', 'const += PEN * (min(d, T - d) < hw)', 'const += PEN * (min(d, T - d) <= hw)'),
    ('tkt_search.py', '                cost += PEN * (np.minimum(d, T - d) < hw)\n            elif lb == li:', '                cost += PEN * (np.minimum(d, T - d) <= hw)\n            elif lb == li:'),
    ('tkt_search.py', 'cost += w * (((other - (oo + base[None, :]) - mt) % T) + mt)', 'cost += w * (((other - (oo + base[None, :])) % T) + mt)'),
    ('tkt_search.py', 'cost += w * ((((oo + base[None, :]) - other - mt) % T) + mt)', 'cost += w * ((((oo + base[None, :]) - other - mt) % T))'),
    ('tkt_search.py', 'const += w * (((d - a - mt) % T) + mt)\n    if hw > 0:', 'const += w * (((d - a) % T) + mt)\n    if hw > 0:'),
    ('tkt_search.py', 'turn_slack(inst.lines[li], inst.T, inst.dwell, inst.tmin) // 2, 0)) for li', 'turn_slack(inst.lines[li], inst.T, inst.dwell, inst.tmin) // 2, 1)) for li'),
    ('tkt_search.py', 'local_search_split(inst, 0 if init else restarts * 3, rng, allowed, init=init, hw=hw)', 'local_search_split(inst, restarts * 3, rng, allowed, init=init, hw=hw)'),
    ('tkt_search.py', 'o2, si2, c2, cache2 = local_search_split(inst, restarts, rng, trial, init=[(o, si)], hw=hw)', 'o2, si2, c2, cache2 = local_search_split(inst, restarts, rng, trial, init=None, hw=hw)'),
    ('tkt_search.py', '            if allowed[li]:\n                continue\n            trial = list(allowed)', '            trial = list(allowed)'),
    ('tkt_exact.py', 'model.Add(v - u + T * p == x)', 'model.Add(v - u + T * p >= x)'),
    ('tkt_exact.py', 'model.Add(sum(cycle_x[li]) <= T * (vmin + fleet_extra))', 'model.Add(sum(cycle_x[li]) <= T * (vmin + fleet_extra + 1))'),
    ('tkt_exact.py', 'activity(ev(la, ea), ev(lb, eb), headway, T - headway,', 'activity(ev(la, ea), ev(lb, eb), headway - 1, T - headway,'),
    ('tkt_exact.py', '        raise ValueError("Zugfolge h muss <= T/2 sein (sonst leere Domäne)")', '        pass'),
    ('tkt_network.py', 'base[i][j] = base[j][i] = 1 + rng.below(10)', 'base[i][j] = base[j][i] = 1 + rng.below(9)'),
    ('tkt_network.py', 'boost = 100 + 6 * hub_pct', 'boost = 100 + 5 * hub_pct'),
    ('tkt_network.py', 'score = GROW * d', 'score = ONE * d'),
    ('tkt_network.py', 'if best_pair is None or best_pair_demand <= 0:', 'if best_pair is None or best_pair_demand < 0:'),
    ('tkt_network.py', 'if sq_dist(coords, best_ext, line[0]) <= sq_dist(coords, best_ext, line[-1]):', 'if sq_dist(coords, best_ext, line[0]) >= sq_dist(coords, best_ext, line[-1]):'),
    ('tkt_network.py', '(isqrt(sq_dist(coords, a, b)) + 2) // 5', '(isqrt(sq_dist(coords, a, b)) + 3) // 5'),
    ('tkt_rng.py', 'z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK', 'z = ((z ^ (z >> 29)) * 0xBF58476D1CE4E5B9) & _MASK'),
    ('tkt_evaluation.py', '    if gain > threshold:', '    if gain >= threshold:'),
    ('tkt_evaluation.py', '    if gain > 0:\n        return "noise"', '    if gain >= 0:\n        return "noise"'),
    ('tkt_evaluation.py', 'step = curve[min(budget, n)]', 'step = curve[0]'),
    ('tkt_evaluation.py', 'return sorted(reciprocal_pairs(inst), key=lambda p: -p[2])', 'return sorted(reciprocal_pairs(inst), key=lambda p: p[2])'),
    ('tkt_evaluation.py', 'return 0, slack, slack // 2', 'return 0, slack, slack // 3'),
]


def check_unique():
    bad = []
    for n, (name, old, new) in enumerate(MUTANTS, 1):
        text = (ROOT / name).read_bytes().decode("utf-8").replace("\r\n", "\n")
        if text.count(old) != 1:
            bad.append((n, name, old[:70], text.count(old)))
        if old == new:
            bad.append((n, name, "alt == neu", 0))
    return bad


def run_one(args):
    n, name, old, new, base = args
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"tkt_mut{n}_"))
    try:
        shutil.copytree(base, tmp, dirs_exist_ok=True)
        path = tmp / name
        original = path.read_bytes().decode("utf-8")
        path.write_bytes(original.replace(old, new).encode("utf-8"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        files = [f"tests/{f}" for f in TEST_ORDER + (["test_app.py"] if WITH_APP else [])]
        try:
            r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *files], cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
            return n, name, old, new, r.returncode == 0, False
        except subprocess.TimeoutExpired:
            return n, name, old, new, False, True                 # Endlosschleife oder zu langsam: gilt als gefunden
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else ""
    jobs = 6
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
        only = "" if only == str(jobs) else only
    wanted = None
    if "--indices" in sys.argv:
        spec = sys.argv[sys.argv.index("--indices") + 1]
        only = "" if only == spec else only
        wanted = set()
        for part in spec.split(","):
            lo, _, hi = part.partition("-")
            wanted.update(range(int(lo), int(hi or lo) + 1))
    bad = check_unique()
    for b in bad:
        print("FEHLER (Stelle nicht eindeutig gefunden):", b)
    if "--dry-run" in sys.argv:
        print(f"{len(MUTANTS)} Mutanten, {len(bad)} Fehler in der Mutantenliste")
        return
    base = pathlib.Path(tempfile.mkdtemp(prefix="tkt_mut_base_"))
    for f in ROOT.glob("*.py"):
        (base / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    shutil.copytree(ROOT / "tests", base / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "README.md", base / "README.md")
    shutil.copytree(ROOT / "data", base / "data")
    for f in (base / "tests").glob("*.py"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    bad_ids = {b[0] for b in bad}
    work = [(n, name, old, new, base) for n, (name, old, new) in enumerate(MUTANTS, 1) if n not in bad_ids and (not only or only in name) and (wanted is None or n in wanted)]
    survivors, killed = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, name, old, new, survived, timeout in pool.map(run_one, work):
            if timeout:
                print(f"[{n:3d}] Zeitüberschreitung (als gefunden gezählt)  {name}", flush=True)
            if survived:
                survivors.append((n, name, old[:70], new[:70]))
                print(f"[{n:3d}] ÜBERLEBT  {name}: {old[:70]!r} -> {new[:70]!r}", flush=True)
            else:
                killed += 1
                print(f"[{n:3d}] gefunden  {name}", flush=True)
    print(f"\n{killed} gefunden, {len(survivors)} überlebt, {len(bad)} Fehler in der Mutantenliste")
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
