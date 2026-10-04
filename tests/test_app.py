"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Exakt-Tab, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tkt_constants as C
import tkt_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _has_metric(at, label):
    return any(m.label == label for m in at.metric)


MAIN = "Ø Wartezeit der Umsteiger (beste Stufe)"


def test_default_run_has_no_exception_and_shows_the_four_main_metrics():
    at = _run()
    _ok(at)
    for label in (MAIN, "Umsteiger mit höchstens 10 min", "Zugbestand (Basis + Zusatzzüge)", "Untergrenze durch die Hin/Rück-Kopplung"):
        assert _has_metric(at, label), label
    assert at.session_state["lines_select"] == C.DEFAULT_LINES and at.session_state[PR.budget_key(C.DEFAULT_LINES)] == C.DEFAULT_BUDGET
    assert any("Auf diesem Netz" in s.value for s in list(at.success) + list(at.info))


@pytest.mark.parametrize("name", C.PRESET_ORDER)
def test_every_preset_button_runs_and_sets_the_controls(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["lines_select"] == p["lines"] and at.session_state["headway_select"] == p["headway"]
    assert at.session_state["seed_input"] == p["seed"] and at.session_state[PR.budget_key(p["lines"])] == min(p["budget"], p["lines"])
    assert at.metric and _has_metric(at, MAIN)


@pytest.mark.parametrize("kw", [dict(lines_select=3), dict(lines_select=10), dict(headway_select=3), dict(headway_select=5, lines_select=5),
                                 dict(seed_input=C.SEED_MIN), dict(seed_input=C.SEED_MAX), dict(offset_slider=0), dict(offset_slider=C.T - 1),
                                 dict(view_select=C.VIEW_OPTIONS[0]), dict(view_select=C.VIEW_OPTIONS[2]),
                                 {PR.budget_key(6): 0}, {PR.budget_key(6): 6}])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_more_lines_change_the_main_metric_and_the_budget_slider_bounds():
    six, three = _run(), _run(lines_select=3)
    assert _metric(six, MAIN) != _metric(three, MAIN)
    sl = next(s for s in three.slider if s.key == PR.budget_key(3))
    assert (sl.min, sl.max) == (0, 3) and sl.value == 3
    sl6 = next(s for s in six.slider if s.key == PR.budget_key(6))
    assert (sl6.min, sl6.max) == (0, 6) and sl6.value == C.DEFAULT_BUDGET


def test_switching_the_line_count_keeps_each_budget_slider_valid():
    at = _run()
    at.slider(key=PR.budget_key(6)).set_value(5).run()
    at.select_slider(key="lines_select").set_value(3).run()
    _ok(at)
    assert next(s for s in at.slider if s.key == PR.budget_key(3)).max == 3
    at.select_slider(key="lines_select").set_value(8).run()
    _ok(at)
    assert next(s for s in at.slider if s.key == PR.budget_key(8)).max == 8


def test_budget_zero_shows_no_additional_trains():
    at = _run(**{PR.budget_key(6): 0})
    _ok(at)
    assert _metric(at, "Zugbestand (Basis + Zusatzzüge)") and any(m.delta == "+0 Zusatzzüge" for m in at.metric if m.label == "Zugbestand (Basis + Zusatzzüge)")


def test_headway_adds_the_cost_message_and_changes_the_table():
    off, on = _run(lines_select=5), _run(lines_select=5, headway_select=5)
    assert not any("Mindest-Zugfolge von" in x.value for x in list(off.info) + list(off.warning))
    assert any("Mindest-Zugfolge von 5 min" in x.value for x in list(on.info) + list(on.warning))
    assert any("verletzte Abschnittspaare" in " ".join(map(str, df.value.columns)) for df in on.dataframe)
    assert not any("verletzte Abschnittspaare" in " ".join(map(str, df.value.columns)) for df in off.dataframe[:1])


def test_dice_button_changes_the_seed_and_the_network():
    at = _run()
    old_seed, old = at.session_state["seed_input"], _metric(at, MAIN)
    next(b for b in at.button if b.label == "🎲 Neues Netz würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed and _metric(at, MAIN) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["lines"] = "7"
    at.query_params["hw"] = "9"
    at.query_params["budget"] = "99"
    at.query_params["seed"] = "99999"
    at.run()
    _ok(at)
    assert at.session_state["lines_select"] == 6 and at.session_state["headway_select"] == 5
    assert at.session_state[PR.budget_key(6)] == 6 and at.session_state["seed_input"] == C.SEED_MAX


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["lines"] = "viele"
    at.query_params["seed"] = "x"
    at.query_params["view"] = "unbekannt"
    at.query_params["budget"] = "?"
    at.run()
    _ok(at)
    assert at.session_state["lines_select"] == C.DEFAULT_LINES and at.session_state["seed_input"] == C.DEFAULT_SEED
    assert at.session_state["view_select"] == C.DEFAULT_VIEW


def test_permalink_roundtrip_keeps_the_configuration():
    at = _run()
    next(b for b in at.button if b.key == "preset_Enge Zugfolge").click().run()
    qp = at.query_params
    at2 = AppTest.from_file(APP, default_timeout=300)
    for k in ("lines", "hw", "budget", "seed", "view"):
        at2.query_params[k] = qp[k]
    at2.run()
    _ok(at2)
    p = C.PRESETS["Enge Zugfolge"]
    assert at2.session_state["lines_select"] == p["lines"] and at2.session_state["headway_select"] == p["headway"] and at2.session_state["seed_input"] == p["seed"]


def test_sections_expanders_and_charts_are_present():
    at = _run()
    _ok(at)
    headers = [s.value for s in at.subheader] + [m.value for m in at.markdown]
    assert any("Hin und zurück sind gekoppelt" in h for h in headers) and any("Was die Messreihe über 150 Netze zeigt" in h for h in headers)
    assert any("Wie lange wartet, wer umsteigt" in h for h in headers)
    titles = [e.label for e in at.expander]
    assert "🔧 Wie wir das erreichen – vollständiger Methodenvergleich" in titles and "Wie funktioniert diese Demo?" in titles and "📐 Mathematische Formulierung" in titles
    assert [t.label for t in at.tabs] == ["🗺️ Fahrplan", "🚆 Zusatzzüge", "🚦 Zugfolge", "🧮 Exakt (OR-Tools)", "📈 Messreihe"]
    assert len(at.get("plotly_chart")) >= 7


def test_coupling_section_shows_two_values_and_a_verdict():
    at = _run()
    _ok(at)
    low = int(_metric(at, "Kleinste Summe").split()[0])
    total = int(_metric(at, "Summe beider").split()[0])
    assert total in (low, low + C.T)
    assert any("Summe liegt" in x.value for x in list(at.success) + list(at.warning))
    at.slider(key="offset_slider").set_value(17).run()
    _ok(at)
    low2 = int(_metric(at, "Kleinste Summe").split()[0])
    assert low2 == low and int(_metric(at, "Summe beider").split()[0]) in (low, low + C.T)      # der Versatz ändert die kleinste Summe nicht, nur welcher Wert gilt
    shift = next(s for s in at.slider if s.key.startswith("shift_slider_"))
    assert shift.max > shift.min
    at.slider(key=shift.key).set_value(shift.min if shift.value != shift.min else shift.max).run()
    _ok(at)
    assert int(_metric(at, "Kleinste Summe").split()[0]) != low                              # die Standzeit verschieben ändert c und damit die kleinste Summe


def test_exact_tab_runs_cpsat_on_the_current_network():
    at = _run(lines_select=3)
    _ok(at)
    next(b for b in at.button if b.key == "exact_button").click().run()
    _ok(at)
    assert _has_metric(at, "Fund (Ø Wartezeit)") and _has_metric(at, "Schranke")
    fund, bound = float(_metric(at, "Fund (Ø Wartezeit)").split()[0]), float(_metric(at, "Schranke").split()[0])
    assert bound <= fund + 1e-9
    at.select_slider(key="lines_select").set_value(5).run()                  # anderes Netz: der alte Lauf wird nicht angezeigt
    assert not _has_metric(at, "Fund (Ø Wartezeit)") and any("erneut lösen" in c.value for c in at.caption)


def test_seed_control_uses_the_portfolio_wording():
    at = _run()
    assert [n.label for n in at.number_input] == ["Zufalls-Seed"]


def test_related_demos_are_linked_and_footer_is_present():
    at = _run()
    text = " ".join(c.value for c in at.caption)
    for name in ("transit-demo", "raptor-demo"):
        assert name in text
    assert "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in text
    assert "geplant" not in text and "noch nicht" not in text
