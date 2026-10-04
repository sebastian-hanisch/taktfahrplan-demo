"""Feste Annahmen, Regler-Grenzen, Presets und Farben der Taktfahrplan-Demo (eine Wahrheitsquelle)."""

# ------------------------------------------------------------------ Fahrplanmodell (feste Annahmen)
T = 60                      # Takt in Minuten
N_STOPS = 20                # Haltestellen im synthetischen Netz
MAX_LINE_LENGTH = 6         # höchstens so viele Halte je Linie
DWELL = 2                   # Halt je Haltestelle (min); im exakten Modell wahlweise 1-3 min
TURN_MIN = 4                # kürzeste Standzeit an einem Endpunkt (min)
TRANSFER_MIN = 3            # Mindest-Umsteigezeit (min)
MIN_TRANSFERS = 3           # ein Seed gilt erst mit mindestens so vielen Umsteigeverbindungen als gültig
SEED_ATTEMPTS = 50          # so viele Folge-Seeds probiert `build_valid_instance`

# ------------------------------------------------------------------ Regler
LINES_OPTIONS = (3, 5, 6, 8, 10)
DEFAULT_LINES = 6
HEADWAY_OPTIONS = (0, 3, 4, 5)            # 0 = aus
DEFAULT_HEADWAY = 0
BUDGET_MIN = 0
DEFAULT_BUDGET = 3
SEED_MIN, SEED_MAX = 0, 9999
DEFAULT_SEED = 501
VIEW_OPTIONS = ("Phasen optimiert", "+ Aufteilung", "+ Zusatzzug")
DEFAULT_VIEW = VIEW_OPTIONS[1]
EXACT_MODES = ("Phasen (Standard-Standzeit)", "+ Aufteilung der Standzeit (gleiche Züge)",
               "+ Haltezeit 1-3 min (gleiche Züge)", "+ Haltezeit und je Linie ein Zusatzzug")
EXACT_TIME_LIMIT = 20.0

# ------------------------------------------------------------------ Suche
SEARCH_RESTARTS = 40        # Neustarts der Lokalsuche (Messung AP 0: erst ab 40 gleichauf mit dem CP-SAT-Fund)
PHASE_RESTARTS = 20         # Neustarts des reinen Phasen-Abstiegs
CURVE_RESTARTS_LIVE = 1     # Neustarts je Kurvenschritt live (Vollmessung: 6)
CURVE_RESTARTS_SWEEP = 6
RANDOM_SAMPLES = 300        # zufällige Phasen: Mittel über so viele Ziehungen

# ------------------------------------------------------------------ Ergebnisdatei
RESULTS_FILE = "data/tkt_results.json"
SWEEP_LINES = (3, 5, 6, 8, 10)
SWEEP_NETS = 30
SWEEP_SEED_START = {3: 100, 5: 200, 6: 0, 8: 300, 10: 400}     # Seeds der Messreihe; Presets liegen bewusst außerhalb
NOISE_RUNS = 5

# ------------------------------------------------------------------ Darstellung
STAGE_COLORS = {"same": "#7d8898", "random": "#95a0ae", "greedy": "#5b8fc9", "phase": "#2a6fb0",
                "split": "#2e7d4f", "add": "#c77700"}
LINE_COLORS = ["#2a6fb0", "#2e7d4f", "#c77700", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#d35400", "#2c3e50", "#b03a6f"]

# ------------------------------------------------------------------ Presets (Seeds liegen außerhalb der Messreihen-Seeds)
PRESET_ORDER = ["Standard", "Dichtes Netz", "Kleines Netz", "Enge Zugfolge", "Viele Zusatzzüge"]
PRESETS = {
    "Standard": {"lines": 6, "headway": 0, "budget": 3, "seed": 501},
    "Dichtes Netz": {"lines": 10, "headway": 0, "budget": 3, "seed": 528},
    "Kleines Netz": {"lines": 3, "headway": 0, "budget": 1, "seed": 535},
    "Enge Zugfolge": {"lines": 8, "headway": 5, "budget": 3, "seed": 510},
    "Viele Zusatzzüge": {"lines": 6, "headway": 0, "budget": 6, "seed": 517},
}
PRESET_HELP = {
    "Standard": "6 Linien, keine Mindest-Zugfolge: der Grundfall. Selbst im besten Takt warten viele Umsteiger länger als 10 Minuten.",
    "Dichtes Netz": "10 Linien: mehr Linien, nicht mehr Wartezeit - die Dichte des Netzes ist nicht das Problem.",
    "Kleines Netz": "3 Linien: schon hier entscheidet die Kopplung von Hin- und Rückrichtung, nicht die Vernetzung.",
    "Enge Zugfolge": "8 Linien, mindestens 5 Minuten zwischen Zügen auf demselben Abschnitt: kostet Wartezeit, die Aufteilung fängt einen Teil auf.",
    "Viele Zusatzzüge": "Alle 6 Linien dürfen einen Zusatzzug einsetzen: der Ertrag sättigt schon nach etwa drei Zügen.",
}


def fmt_min(x, digits=1):
    return f"{x:.{digits}f} min"


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}"


def fmt_pct(x, digits=0):
    return f"{100 * x:.{digits}f} %"
