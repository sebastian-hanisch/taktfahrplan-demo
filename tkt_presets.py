"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Der Regler der Zusatzzug-Erlaubnis hat Grenzen, die von der Linienzahl abhängen (0 bis Linienzahl). Streamlit setzt einen
Regler auf den Mindestwert zurück, wenn sich `max_value` bei gleichem Key ändert - deshalb trägt der Key die Linienzahl
(`budget_slider_<Linien>`), und sein Zustand wird nur in dem Lauf gesetzt, in dem er auch gezeichnet wird (Callback oder
`seed_widget` unmittelbar vor `st.slider`).
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import tkt_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "lines_select": SettingSpec("lines", int, C.DEFAULT_LINES, C.LINES_OPTIONS[0], C.LINES_OPTIONS[-1]),
    "headway_select": SettingSpec("hw", int, C.DEFAULT_HEADWAY, C.HEADWAY_OPTIONS[0], C.HEADWAY_OPTIONS[-1]),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, C.SEED_MIN, C.SEED_MAX),
    "view_select": SettingSpec("view", str, C.DEFAULT_VIEW),
}
BUDGET_URL = "budget"


def budget_key(lines):
    return f"budget_slider_{int(lines)}"


def budget_bounds(lines):
    return C.BUDGET_MIN, int(lines)


def default_budget(lines):
    return min(C.DEFAULT_BUDGET, int(lines))


def seed_budget_widget(lines):
    """Unmittelbar vor `st.slider(key=budget_key(lines))` aufrufen: legt den Startwert in dem Lauf an, in dem der Regler gezeichnet wird."""
    key = budget_key(lines)
    if key not in st.session_state:
        st.session_state[key] = default_budget(lines)
    return key


def snap(options, value):
    """Regler mit festen Stufen: ein Permalink-Wert dazwischen rastet auf die nächste Stufe ein (bei Gleichstand auf die kleinere)."""
    return min(options, key=lambda o: (abs(o - value), o))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clamp(spec, value):
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.caster is int:
                    value = _clamp(spec, value)
                if state_key == "lines_select":
                    value = snap(C.LINES_OPTIONS, value)
                elif state_key == "headway_select":
                    value = snap(C.HEADWAY_OPTIONS, value)
                elif state_key == "view_select" and value not in C.VIEW_OPTIONS:
                    continue
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    if BUDGET_URL in qp:
        try:
            lines = st.session_state.get("lines_select", C.DEFAULT_LINES)
            lo, hi = budget_bounds(lines)
            st.session_state[budget_key(lines)] = max(lo, min(hi, int(qp[BUDGET_URL])))
        except (ValueError, TypeError):
            pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values, budget):
    """`values`: {state_key: aktueller Wert}; `budget`: aktuelle Zusatzzug-Erlaubnis."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
        st.query_params[BUDGET_URL] = str(budget)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["lines_select"] = p["lines"]
    st.session_state["headway_select"] = p["headway"]
    st.session_state["seed_input"] = p["seed"]
    st.session_state[budget_key(p["lines"])] = min(p["budget"], p["lines"])


def randomize_seed():
    st.session_state["seed_input"] = random.randint(C.SEED_MIN, C.SEED_MAX)
