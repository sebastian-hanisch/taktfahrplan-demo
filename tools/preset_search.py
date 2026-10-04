"""Sucht für jedes Preset Seeds außerhalb der Messreihen-Seeds, bei denen alle Abnahmekriterien (tkt_stories) gelten,
und schreibt den Messbericht nach tools/PRESET_SWEEP.md. Braucht data/tkt_results.json.

  python tools/preset_search.py [Seeds je Preset (Standard 40)]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tkt_constants as C  # noqa: E402
import tkt_results as R  # noqa: E402
import tkt_stories as S  # noqa: E402
from tkt_evaluation import run_live  # noqa: E402

START = 500          # hinter allen Messreihen-Seeds (die liegen unter 430)


def evaluate(args):
    name, seed = args
    p = C.PRESETS[name]
    res = R.load_results()
    run = run_live(p["lines"], seed, p["headway"], p["budget"])
    facts = S.facts_for(run, res, R.noise_thresholds(res))
    return name, seed, run["seed"], S.check(name, facts), facts


def typical_target(name, res):
    """(Kennzahl, Mittelwert der Messreihe) für die Auswahl des typischsten Seeds."""
    p = C.PRESETS[name]
    if name == "Enge Zugfolge":
        row = next(r for r in R.headway_rows(res) if r["lines"] == p["lines"] and r["h"] == p["headway"])
        return "headway_cost", row["cost"]
    if name in ("Kleines Netz", "Viele Zusatzzüge"):
        return "add_gain", R.paired(res, p["lines"])["add_gain"]["mean"]
    return "split_gain", R.paired(res, p["lines"])["split_gain"]["mean"]


def main():
    res = R.load_results()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = [(name, seed) for name in C.PRESET_ORDER for seed in range(START, START + n)]
    results = {name: [] for name in C.PRESET_ORDER}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for name, seed, used, checks, facts in ex.map(evaluate, jobs):
            results[name].append((seed, used, checks, facts))
    lines = ["# Preset-Abstimmung (tools/preset_search.py)", "",
             f"Seeds {START}..{START + n - 1} je Preset, außerhalb der Messreihen-Seeds ({', '.join(f'{k}: {v}-{v + C.SWEEP_NETS - 1}' for k, v in C.SWEEP_SEED_START.items())}). "
             "Ein Seed besteht, wenn alle Kriterien des Presets gelten (siehe `tkt_stories.py`; gleiche Kriterien prüft `tests/test_claims.py` an den gewählten Seeds).", ""]
    chosen = {}
    for name in C.PRESET_ORDER:
        rows = results[name]
        ok = [(seed, used, facts) for seed, used, checks, facts in rows if all(c for _, _, c in checks)]
        lines += [f"## {name}", "", "Kriterien: " + "; ".join(f"{cid} ({text})" for cid, text, _ in rows[0][2]), "",
                  f"Bestanden: {len(ok)} von {len(rows)} Seeds.", ""]
        if ok:
            # unter den bestandenen: der typischste, gemessen an der Kennzahl, die die Geschichte des Presets trägt, gegen das Mittel der Messreihe; dann der kleinste Seed
            key, target = typical_target(name, res)
            ok.sort(key=lambda t: (abs(t[2][key] - target), t[0]))
            chosen[name] = ok[0][0]
            lines += [f"Gewählt: Seed {ok[0][0]} (tatsächlicher Seed {ok[0][1]}), Aufteilungs-Gewinn {ok[0][2]['split_gain']:.2f} min, Zusatzzug-Gewinn {ok[0][2]['add_gain']:.2f} min, "
                      f"Zugfolge-Mehrkosten {ok[0][2]['headway_cost']:.2f} min.", ""]
        else:
            lines += ["**Kein Seed besteht alle Kriterien.**", ""]
            fails = {}
            for _, _, checks, _ in rows:
                for cid, _, good in checks:
                    fails[cid] = fails.get(cid, 0) + (not good)
            lines += ["Durchgefallen je Kriterium: " + ", ".join(f"{k} {v}x" for k, v in fails.items()), ""]
    (ROOT / "tools" / "PRESET_SWEEP.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print({k: v for k, v in chosen.items()})
    print("geschrieben: tools/PRESET_SWEEP.md")


if __name__ == "__main__":
    main()
