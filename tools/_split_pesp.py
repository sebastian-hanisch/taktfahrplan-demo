"""Einmalig: pesp.py (messreihe_pesp) in tkt_timetable / tkt_search / tkt_exact zerlegen (Zeilenbereiche)."""
import pathlib
src = pathlib.Path(r"C:\Users\sebas\Claude Code Projects\schiene-planung\messreihe_pesp\pesp.py").read_text(encoding="utf-8").split("\n")
def L(a, b):  # 1-basiert, einschließlich
    return "\n".join(src[a - 1:b]) + "\n"
root = pathlib.Path(r"C:\Users\sebas\Claude Code Projects\taktfahrplan-demo")

timetable = '''"""Fahrplanmodell: Linien, Ereigniszeiten im Takt, Umsteiger, Wartezeiten, Hin/Rück-Kopplung (Untergrenze).

Jede Linie fährt in beide Richtungen im Takt T. Ereignisse = Ankunft/Abfahrt je Haltestelle und Richtung; Umstieg =
Aktivität Ankunft -> Abfahrt mit Spannung x = pi_j - pi_i + T*p in [m, m+T-1]. Kleine Einheiten mit expliziten Ein- und
Ausgaben (DEMO-PLAYBOOK 1b); kein Zufall, kein Zustand.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

T_DEFAULT = 60

''' + L(24, 39) + "\n" + L(41, 74) + "\n" + L(76, 79) + "\n" + L(81, 149) + "\n" + L(151, 153) + "\n" + L(234, 250) + "\n" + L(468, 470) + "\n" + L(505, 534)
(root / "tkt_timetable.py").write_text(timetable.rstrip("\n") + "\n", encoding="utf-8", newline="\n")

search = '''"""Lokalsuche für den Takt: Greedy, Phasen-Abstieg, Aufteilung der Wende-Standzeit, Zusatzzug, Mindest-Zugfolge.

Die Kandidatenmatrix `line_cost_matrix` bewertet für eine Linie alle (Phase, Aufteilung)-Paare auf einmal (numpy,
ohne Zufall); Zugfolge ist eine harte Bedingung per Strafterm PEN. Zufall nur über einen übergebenen Generator mit
`randrange` (SplitMix64 aus tkt_rng).
"""
from __future__ import annotations

import numpy as np

from tkt_timetable import Instance, Line, base_fleet, fixed_line_times, shared_segments, turn_slack

''' + L(156, 231) + "\n" + L(345, 466) + "\n" + L(471, 474) + "\n" + L(535, 541)
(root / "tkt_search.py").write_text(search.rstrip("\n") + "\n", encoding="utf-8", newline="\n")

exact = '''"""Exakte Referenz: das volle PESP-Modell in OR-Tools CP-SAT (Phasen, Wende-Spielraum, optional Haltezeit-Spielraum,
Zugfolge, Fahrzeugdeckel). Liefert Fund, Schranke und Status; beweist beim erweiterten Modell selten."""
from __future__ import annotations

import math
import os
from collections import defaultdict

from ortools.sat.python import cp_model

from tkt_timetable import Instance, cycle_min, fixed_line_times, shared_segments, turn_slack

''' + L(253, 342)
(root / "tkt_exact.py").write_text(exact.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
print("ok")
