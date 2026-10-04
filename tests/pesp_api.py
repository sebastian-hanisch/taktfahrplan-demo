"""Ein Namensraum `P` über Fahrplanmodell, Suche und CP-SAT-Referenz, damit die Mini-Instanz-Tests nach den Funktionsnamen lesbar bleiben."""
from types import SimpleNamespace

import tkt_exact
import tkt_search
import tkt_timetable

P = SimpleNamespace(**{name: getattr(mod, name) for mod in (tkt_timetable, tkt_search, tkt_exact) for name in dir(mod) if not name.startswith("_")})
