"""Taktfahrplan: Umsteigen im Takt (PESP) - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fortsetzung der Liniennetz-Demo (transit-demo, dort ausdrücklich "ohne Taktfrequenz/Fahrplan") und erster Baustein der Reihe
Bahn/Schienenverkehr: Jede Linie fährt in beide Richtungen alle 60 Minuten, gewählt wird, in welcher Minute sie startet. Die Demo
zeigt, wie lange Umsteiger wartend im Takt stehen, warum Hin- und Rückrichtung gekoppelt sind und was Standzeit am Endpunkt,
Zusatzzüge und eine Mindest-Zugfolge verändern. Modell: Periodic Event Scheduling Problem (Serafini & Ukovich 1989).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import tkt_constants as C
import tkt_results as R
from tkt_evaluation import (STAGE_LABELS, STAGE_ORDER, VIEW_TO_STAGE, headway_message, pair_curves, pair_label, pair_shift_range,
                            run_exact, run_live, sorted_pairs, split_message, top_pairs_table, view_schedule)
from tkt_pdf_export import generate_timetable_pdf
from tkt_presets import (apply_preset, bounds, budget_bounds, init_session_state_defaults, load_permalink_settings,
                         randomize_seed, seed_budget_widget, sync_query_params)
from tkt_visualization import (build_cascade_chart, build_coupling_chart, build_curve_chart, build_headway_chart, build_network_map,
                               build_size_chart, build_staircase)

st.set_page_config(page_title="Taktfahrplan – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _results():
    return R.load_results()


@st.cache_data(show_spinner=False)
def _live(lines, seed, headway, budget):
    return run_live(lines, seed, headway, budget)


def _hw_label(h):
    return "aus" if h == 0 else f"{h} min"


st.title("🚆 Taktfahrplan: Umsteigen im Takt")
st.markdown(
    """
Ein Liniennetz steht, jetzt braucht es einen **Takt**: Jede Linie fährt in beide Richtungen alle 60 Minuten, und man wählt, in
welcher Minute sie startet. Wer **umsteigt**, wartet je nachdem wenige Minuten oder fast eine Stunde. Die Demo zeigt, wie viel der
Takt spart, warum selbst der beste Takt vielen Umsteigern keinen kurzen Anschluss geben kann (**Hin- und Rückrichtung sind
gekoppelt**) und was **Standzeit am Endpunkt**, **Zusatzzüge** und eine **Mindest-Zugfolge** verändern - live auf einem Netz und
vorgerechnet über 150 Netze. Mehr dazu in „Wie funktioniert diese Demo?“ und „📐 Mathematische Formulierung“ am Ende der Seite.
"""
)
st.caption(
    "Fortsetzung von [transit-demo](https://sebastianhanisch-transit-demo.streamlit.app/) (Liniennetz ohne Fahrplan) und erster Baustein "
    "der Reihe Bahn/Schienenverkehr. Anders als [raptor-demo](https://sebastianhanisch-raptor-demo.streamlit.app/) (Verbindungsauskunft auf "
    "einem gegebenen Fahrplan) werden hier die Abfahrtsminuten der Linien gewählt."
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(3)
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i % 3]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    lines = st.select_slider("Linien", options=C.LINES_OPTIONS, key="lines_select",
                             help="Mehr Linien = mehr Umsteigeverbindungen und mehr gemeinsam befahrene Abschnitte, und eine längere Rechenzeit.")
    headway = st.select_slider("Mindest-Zugfolge", options=C.HEADWAY_OPTIONS, key="headway_select", format_func=_hw_label,
                               help="Auf jedem Abschnitt, den zwei Linien gemeinsam befahren, liegen die Abfahrten mindestens so viele Minuten "
                                    "auseinander. Wirkt als harte Bedingung auf alle optimierten Stufen.")
    lo_b, hi_b = budget_bounds(lines)
    budget = st.slider("Zusatzzug-Erlaubnis (Linien)", lo_b, hi_b, key=seed_budget_widget(lines),
                       help="Wie viele Linien einen Zusatzzug einsetzen dürfen (jeweils die Linien, bei denen es am meisten bringt). "
                            "Mit einem Zusatzzug darf die Standzeit am Endpunkt um einen ganzen Takt länger werden.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input",
                           help="Bestimmt Haltestellen, Nachfrage und Linien. Der Seed zählt hoch, bis das Netz mindestens "
                                f"{C.MIN_TRANSFERS} Umsteigeverbindungen hat.")
    st.button("🎲 Neues Netz würfeln", on_click=randomize_seed)

lines, headway, budget, seed = int(lines), int(headway), int(budget), int(seed)
sync_query_params({"lines_select": lines, "headway_select": headway, "seed_input": seed, "view_select": st.session_state["view_select"]}, budget)

res = _results()
noise = R.noise_thresholds(res)
with st.spinner(f"Rechne alle Stufen für {lines} Linien (Lokalsuche mit {C.SEARCH_RESTARTS} Neustarts, Kurve der Zusatzzüge) …"):
    run = _live(lines, seed, headway, budget)
inst, stages, coords = run["inst"], run["stages"], run["coords"]
n_lines = len(inst.lines)
st.caption(
    f"Netz aus Seed {run['seed']}{' (der eingegebene Seed hatte zu wenige Umsteigeverbindungen)' if run['seed'] != seed else ''}: {C.N_STOPS} Haltestellen, "
    f"{n_lines} Linien, {run['n_pairs']} Anschlüsse mit Gegenrichtung, Zugbestand {run['base_fleet']} ohne Zusatzzüge. "
    f"Rechenzeit dieses Laufs {run['seconds']:.1f} s (beim ersten Aufruf; danach aus dem Zwischenspeicher)."
)

st.markdown("---")
st.markdown("## 🚆 Wie lange wartet, wer umsteigt – und was kostet jede Verbesserung?")
best = stages["add"]
m1, m2 = st.columns(2)
m3, m4 = st.columns(2)
m1.metric("Ø Wartezeit der Umsteiger (beste Stufe)", f"{best['avg']:.1f} min", delta=f"{best['avg'] - stages['random']['avg']:+.1f} min gegen zufällige Phasen",
          delta_color="inverse", help="Gewichtet mit der Nachfrage; enthält die Mindest-Umsteigezeit von 3 min. Ideal wären 3 min.")
m2.metric("Umsteiger mit höchstens 10 min", f"{100 * best['le10']:.0f} %", delta=f"{100 * (best['le10'] - stages['phase']['le10']):+.0f} Pp. gegen nur Phasen",
          help="Anteil der Umsteiger (nachfragegewichtet), die höchstens 10 Minuten warten.")
m3.metric("Zugbestand (Basis + Zusatzzüge)", f"{best['fleet']}", delta=f"{best['fleet'] - run['base_fleet']:+d} Zusatzzüge", delta_color="off",
          help="Zugbestand = Summe der Fahrzeuge aller Linien (Zyklusdauer geteilt durch den Takt).")
m4.metric("Untergrenze durch die Hin/Rück-Kopplung", f"{run['lower_bound']:.1f} min",
          help="Mit Standard-Standzeit kann kein Takt im Mittel besser sein (Phasen-Modell). Die Aufteilung der Standzeit bricht diese Grenze.")

thr_split = noise["split"]
gain, (state, text) = split_message(stages, thr_split)
(st.success if state == "over" else st.info)(f"💡 {text}")
hmsg = headway_message(stages, run["free_split"], headway, thr_split)
if hmsg:
    cost, hstate, htext = hmsg
    (st.warning if hstate == "over" else st.info)(f"🚦 {htext}")
if not stages["split"].get("feasible", True) or stages["add"]["violations"]:
    st.error("Für diese Zugfolge hat die Suche keinen zulässigen Fahrplan gefunden; die Stufen unten zeigen die verletzten Abschnittspaare.")

cmp_lines = R.nearest_lines(lines)
cascade = R.cascade(res, cmp_lines)
if headway == 0:
    reference = {**cascade, "add": R.add_stage_at(res, cmp_lines, budget)}
    ref_note = f"Messreihe: Ø über {R.paired(res, cmp_lines)['n']} Netze mit {cmp_lines} Linien; „+ Zusatzzug“ bei Erlaubnis für {min(budget, cmp_lines)} Linien."
else:
    reference = {k: cascade[k] for k in ("same", "random", "greedy")}
    for k, stage_key in (("phase", "phase"), ("split", "split")):
        vals = [r["hw"][str(headway)][stage_key]["avg"] for r in R.nets(res, cmp_lines) if r["hw"][str(headway)][stage_key]["avg"] is not None]
        mean, se = R.mean_se(vals)
        reference[k] = {"mean": mean, "se": se}
    reference["add"] = {"mean": float("nan"), "se": float("nan")}
    ref_note = (f"Messreihe: Ø über {R.paired(res, cmp_lines)['n']} Netze mit {cmp_lines} Linien und {headway} min Zugfolge für Phasen und Aufteilung; "
                "für den Zusatzzug gibt es bei Zugfolge keine Vergleichszahl für diese Erlaubnis.")
st.plotly_chart(build_staircase(stages, run["lower_bound"], reference, f"Messreihe Ø ({cmp_lines} Linien)"), width="stretch",
                key=f"staircase_{lines}_{seed}_{headway}_{budget}")
st.caption(f"Balken: dieses Netz. Punkte mit Fehlerbalken: {ref_note} Ein einzelnes Netz streut (Spannweite zweier Läufe bis etwa "
           f"{noise['split']:.1f} min); die Mittelwerte sind die Aussage.")

rows_tbl = []
for k in STAGE_ORDER:
    s = stages[k]
    rows_tbl.append({"Stufe": STAGE_LABELS[k], "Ø Wartezeit (min)": round(s["avg"], 1), "Umsteiger ≤ 10 min": f"{100 * s['le10']:.0f} %",
                     "Zugbestand": s["fleet"], **({"verletzte Abschnittspaare": "–" if s["violations"] is None else s["violations"]} if headway else {})})
st.dataframe(pd.DataFrame(rows_tbl), width="stretch", hide_index=True)
if headway:
    st.caption(f"Bei {headway} min Zugfolge zählen die verletzten Abschnittspaare der Referenzstufen (zur vollen Stunde, Greedy); die optimierten Stufen halten die Bedingung ein. "
               f"Es gibt {len(run['pairs_segments'])} Abschnittspaare auf diesem Netz.")

st.markdown("### Das Netz und sein Fahrplan")
view = st.radio("Fahrplan anzeigen für", C.VIEW_OPTIONS, key="view_select", horizontal=True,
                help="Reine Anzeigewahl: welche Stufe als Netzkarte und Abfahrtstabelle gezeigt wird (keine Einstellung der Rechnung).")
vstage = stages[VIEW_TO_STAGE[view]]
col_map, col_tab = st.columns([3, 2])
with col_map:
    st.plotly_chart(build_network_map(inst, coords, allowed_lines=stages["add"].get("allowed_lines") if VIEW_TO_STAGE[view] == "add" else None),
                    width="stretch", key=f"map_{lines}_{seed}_{view}")
with col_tab:
    st.markdown("**Abfahrtsminuten je Linie** (jede Stunde, Hin- / Rückrichtung)")
    for li, rows in view_schedule(inst, vstage):
        df = pd.DataFrame([{"Halt": s, "hin": "–" if h is None else f"{h:02d}", "zurück": "–" if r is None else f"{r:02d}"} for s, h, r in rows])
        with st.expander(f"Linie {li + 1}: {' – '.join(str(s) for s, _, _ in rows)}", expanded=(li == 0)):
            st.dataframe(df, width="stretch", hide_index=True)
summary_lines = [f"Netz aus Seed {run['seed']}, {n_lines} Linien, Takt {C.T} min, Stufe: {view}",
                 f"Mittlere Wartezeit der Umsteiger {vstage['avg']:.1f} min, {100 * vstage['le10']:.0f} % warten höchstens 10 min, Zugbestand {vstage['fleet']}."]
st.download_button("📄 Fahrplanaushang als PDF herunterladen", data=generate_timetable_pdf(inst, vstage, f"Taktfahrplan - {view}", summary_lines),
                   file_name="taktfahrplan.pdf", mime="application/pdf", key="pdf_download")

# ------------------------------------------------------------------ Kernabschnitt: Hin und zurück sind gekoppelt
st.markdown("---")
st.subheader("🔗 Hin und zurück sind gekoppelt")
st.markdown(
    "Wer an einem Anschluss in beide Richtungen umsteigt, wartet hin und zurück zusammen **mindestens 2·3 + c Minuten**, egal wie man die Linien "
    "verschiebt: schiebt man die eine Wartezeit kürzer, wird die andere um genau so viel länger. Die Summe nimmt nur **zwei Werte** an (der kleinere gilt für "
    "genau c + 1 der 60 Versätze, sonst einen Takt mehr). Probieren Sie es an einem echten Anschluss dieses Netzes:"
)
pairs = sorted_pairs(inst)
if not pairs:
    st.info("Dieses Netz hat keinen Anschluss mit Gegenrichtung - es gibt nichts zu koppeln.")
else:
    net_id = f"{lines}_{run['seed']}"
    pi = st.selectbox("Anschluss", list(range(len(pairs))), key=f"pair_select_{net_id}", format_func=lambda i: pair_label(inst, pairs[i]),
                      help="Die Anschlüsse des Netzes, nach Gewicht sortiert. Linie 1 des Paares liegt bei Phase 0, Linie 2 wird verschoben.")
    pair = pairs[pi]
    lb = pair[0][3]
    lo_s, hi_s, std_s = pair_shift_range(inst, pair)
    c_off, c_shift = st.columns(2)
    with c_off:
        offset = st.slider("Versatz der zweiten Linie (min)", 0, C.T - 1, key="offset_slider", help="Minute der ersten Abfahrt der zweiten Linie gegen die erste.")
    split_b = None
    with c_shift:
        if hi_s > lo_s:
            shift_key = f"shift_slider_{net_id}_{pi}"
            if shift_key not in st.session_state:
                st.session_state[shift_key] = std_s
            s_val = st.slider(f"Standzeit am Endpunkt von Linie {lb + 1} verschieben (min)", lo_s, hi_s, key=shift_key,
                              help="Verteilt den Schlupf zwischen den beiden Endpunkten der zweiten Linie anders (Zugzahl bleibt gleich). "
                                   f"Standard ist die Hälfte ({std_s} min). Das verändert c und damit die kleinste mögliche Summe.")
            split_b = (s_val, 0)
        else:
            st.info(f"Linie {lb + 1} hat keinen Schlupf an den Endpunkten: ihre Standzeit lässt sich nicht verschieben.")
    hin, rueck, sums, c_val, low, n_low = pair_curves(inst, pair, split_b)
    st.plotly_chart(build_coupling_chart(hin, rueck, sums, offset, low), width="stretch", key=f"coupling_{net_id}_{pi}_{split_b}_{offset}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wartezeit Hinrichtung", f"{hin[offset]} min")
    k2.metric("Wartezeit Rückrichtung", f"{rueck[offset]} min")
    k3.metric("Summe beider", f"{sums[offset]} min")
    k4.metric("Kleinste Summe", f"{low} min", help=f"2·3 + c mit c = {c_val}; sie wird bei {n_low} der {C.T} Versätze erreicht (c + 1 = {c_val + 1}).")
    if sums[offset] == low:
        st.success(f"Die Summe liegt auf dem kleineren der beiden Werte ({low} min): besser geht es für diesen Anschluss bei dieser Standzeit nicht.")
    else:
        st.warning(f"Die Summe liegt einen ganzen Takt höher ({sums[offset]} statt {low} min). Der kleinere Wert gilt nur für {n_low} von {C.T} Versätzen"
                   f"{' - oder man verschiebt die Standzeit und ändert damit c' if hi_s > lo_s else ''}.")
    st.caption(f"Die Untergrenze wirkt auf das ganze Netz: im Mittel über {R.paired(res)['n']} Netze der Messreihe erzwingt sie {R.coupling_share(res) * 100:.0f} % des "
               "Aufschlags über die Mindest-Umsteigezeit.")

# ------------------------------------------------------------------ Kernabschnitt: Messreihe
st.markdown("---")
st.subheader("📐 Was die Messreihe über 150 Netze zeigt")
st.caption(f"Vorgerechnet (tools/sweep.py, {res['meta']['nets_per_size']} Netze je Linienzahl, {C.N_STOPS} Haltestellen, Takt {C.T} min): ein einzelnes Netz streut, die Mittelwerte tragen.")
sweep_idx = list(C.LINES_OPTIONS).index(cmp_lines)
sweep_lines = st.selectbox("Netzgröße der Messreihe (Linien)", C.LINES_OPTIONS, index=sweep_idx, key="sweep_lines_select")
casc = R.cascade(res, sweep_lines)
pr = R.paired(res, sweep_lines)
pr_all = R.paired(res)
c_a, c_b = st.columns(2)
with c_a:
    st.markdown(f"**Kaskade bei {sweep_lines} Linien** (Mittel ± Standardfehler)")
    st.plotly_chart(build_cascade_chart(casc, sweep_lines), width="stretch", key=f"sweep_cascade_{sweep_lines}")
with c_b:
    st.markdown("**Netzgröße: die Dichte ist nicht das Problem**")
    st.plotly_chart(build_size_chart(R.size_curves(res)), width="stretch", key="sweep_sizes")
st.dataframe(pd.DataFrame([{"Stufe": STAGE_LABELS[k], "Ø (min)": round(casc[k]["mean"], 1), "± SE": round(casc[k]["se"], 1), "Median": round(casc[k]["median"], 1),
                            "P10": round(casc[k]["p10"], 1), "P90": round(casc[k]["p90"], 1), "Umsteiger ≤ 10 min": f"{100 * casc[k]['le10']:.0f} %",
                            "Zugbestand Ø": round(casc[k]["fleet"], 1)} for k in STAGE_ORDER]), width="stretch", hide_index=True)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Gewinn der Aufteilung", f"{pr['split_gain']['mean']:.1f} ± {pr['split_gain']['se']:.1f} min",
          help=f"Gepaarte Differenz Phasen-Optimum minus Aufteilung je Netz; Median {pr['split_gain']['median']:.1f}, P10 {pr['split_gain']['p10']:.1f}, P90 {pr['split_gain']['p90']:.1f}. "
               f"Positiv in {pr['positive_split']} von {pr['n']} Netzen (alle Größen zusammen: {pr_all['positive_split']} von {pr_all['n']}).")
d2.metric("Gewinn der Zusatzzüge", f"{pr['add_gain']['mean']:.1f} ± {pr['add_gain']['se']:.1f} min",
          help=f"Aufteilung minus Zusatzzug (alle Linien erlaubt); Median {pr['add_gain']['median']:.1f}, P10 {pr['add_gain']['p10']:.1f}, P90 {pr['add_gain']['p90']:.1f}.")
d3.metric("Mehrkosten der Zugfolge (4 min)", f"{pr['headway_cost']['mean']:.1f} ± {pr['headway_cost']['se']:.1f} min",
          help=f"Aufteilung mit 4 min Zugfolge minus ohne; Median {pr['headway_cost']['median']:.1f}, P10 {pr['headway_cost']['p10']:.1f}, P90 {pr['headway_cost']['p90']:.1f}.")
d4.metric("Greedy über dem Phasen-Optimum", f"{pr['greedy_gap']['mean']:.1f} ± {pr['greedy_gap']['se']:.1f} min",
          help="Die Phasen-Aufgabe selbst ist leicht: Greedy ist fast schon optimal. Der Aufhänger ist nicht „Heuristik gegen exakt“.")
st.caption(
    f"Die Hin/Rück-Kopplung erzwingt im Mittel {R.coupling_share(res, sweep_lines) * 100:.0f} % des Aufschlags über die Mindestzeit (alle Größen: {R.coupling_share(res) * 100:.0f} %). "
    f"Phasen-Optimum 3 / 5 / 6 / 8 / 10 Linien: " + " / ".join(f"{R.size_curves(res)['phase'][k]:.1f}" for k in C.LINES_OPTIONS) + " min."
)

# ------------------------------------------------------------------ Methodenvergleich
st.markdown("---")
with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich", expanded=False):
    tabs = st.tabs(["🗺️ Fahrplan", "🚆 Zusatzzüge", "🚦 Zugfolge", "🧮 Exakt (OR-Tools)", "📈 Messreihe"])

    with tabs[0]:
        st.caption(f"Die gewichtigsten Anschlüsse bei der Stufe „{view}“: Wartezeit je Richtung, Summe und die Untergrenze der Summe bei Standard-Standzeit.")
        st.dataframe(pd.DataFrame(top_pairs_table(inst, vstage)), width="stretch", hide_index=True)
        st.caption("Liegt die Summe unter der Untergrenze, hat die Aufteilung der Standzeit die Kopplung gebrochen.")

    with tabs[1]:
        curve = run["curve"]
        st.caption("Wartezeit gegen tatsächlich genutzte Zusatzzüge: die Erlaubnis geht Linie für Linie an die Linie mit dem größten Gewinn.")
        st.plotly_chart(build_curve_chart(curve, run["base_fleet"], R.curve_means(res, cmp_lines) if headway == 0 else None, f"Messreihe Ø ({cmp_lines} Linien)"),
                        width="stretch", key=f"curve_{lines}_{seed}_{headway}")
        st.dataframe(pd.DataFrame([{"Erlaubnis für Linien": e["allowed"], "Ø Wartezeit (min)": round(e["avg"], 1), "Zusatzzüge genutzt": e["fleet"] - run["base_fleet"],
                                    "erlaubte Linien": ", ".join(str(i + 1) for i, a in enumerate(e["allowed_lines"]) if a) or "–"} for e in curve]),
                     width="stretch", hide_index=True)
        st.caption(f"Diese Kurve entsteht live mit {C.CURVE_RESTARTS_LIVE} Neustart je Schritt (schnell, etwas gröber); die Messreihe rechnet mit {res['meta']['curve_restarts']}. "
                   "Je Netz ist die Kurve monoton nicht steigend (Warmstart aus dem vorigen Schritt). Mit Zugfolge gibt es keine Messreihen-Kurve.")

    with tabs[2]:
        st.caption(f"Zugfolge: auf {len(run['pairs_segments'])} Paaren von Abfahrten gemeinsam befahrener Abschnitte dieses Netzes (beide Richtungen) darf der Abstand "
                   "mod 60 nicht unter der Mindest-Zugfolge liegen. Keine Block- oder Weichenlogik - das wäre die Trassenvergabe.")
        if headway:
            st.dataframe(pd.DataFrame([{"Stufe": STAGE_LABELS[k], "verletzte Abschnittspaare": "–" if stages[k]["violations"] is None else stages[k]["violations"]} for k in STAGE_ORDER]),
                         width="stretch", hide_index=True)
        else:
            st.info("Die Mindest-Zugfolge ist aus. In der Seitenleiste auf 3, 4 oder 5 min stellen, um die Bedingung einzurechnen; unten die Messreihe für alle Stufen.")
        hrows = R.headway_rows(res)
        h_sel = st.radio("Mindest-Zugfolge der Messreihe", (3, 4, 5), index=1, key="headway_sweep_select", horizontal=True, format_func=lambda h: f"{h} min")
        st.plotly_chart(build_headway_chart(hrows, h_sel), width="stretch", key=f"headway_sweep_{h_sel}")
        st.dataframe(pd.DataFrame([{"Linien": r["lines"], "Zugfolge": f"{r['h']} min", "Mehrkosten (min)": f"{r['cost']:.1f} ± {r['se']:.1f}", "% darüber": f"{r['pct']:.1f}",
                                    "zulässig gefunden": f"{r['feasible']}/{r['n']}", "Netze, in denen die Lösung ohne Bedingung verletzt": f"{r['nets_violated']}/{r['n']}",
                                    "verletzte Abschnittspaare ohne Bedingung": f"{r['pair_share']:.0f} %"} for r in hrows]), width="stretch", hide_index=True)

    with tabs[3]:
        st.caption(f"Dasselbe Modell exakt in OR-Tools CP-SAT ({C.EXACT_TIME_LIMIT:.0f} s Zeitlimit): liefert einen Fund und eine Schranke. Beim erweiterten Modell wird das Optimum "
                   "selten bewiesen - dann ist der Fund eine gute Lösung, kein bewiesenes Optimum.")
        mode = st.selectbox("Modell", list(range(len(C.EXACT_MODES))), format_func=lambda i: C.EXACT_MODES[i], key="exact_mode_select")
        exact_key = (lines, run["seed"], headway, mode)
        if st.button("🧮 Mit CP-SAT lösen", key="exact_button"):
            with st.spinner(f"CP-SAT rechnet (bis zu {C.EXACT_TIME_LIMIT:.0f} s) …"):
                st.session_state["exact_result"] = {"key": exact_key, "res": run_exact(lines, seed, headway, mode, C.EXACT_TIME_LIMIT)}
        ex = st.session_state.get("exact_result")
        if ex and ex["key"] == exact_key:
            r = ex["res"]
            if "avg" not in r:
                st.error(f"CP-SAT hat keinen Fahrplan gefunden (Status {r['status']}).")
            else:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Fund (Ø Wartezeit)", f"{r['avg']:.1f} min")
                e2.metric("Schranke", f"{r['bound']:.1f} min", help="Bessere Fahrpläne als dieser Wert gibt es nicht.")
                e3.metric("Status", "bewiesen optimal" if r["status"] == "OPTIMAL" else "Fund, nicht bewiesen")
                e4.metric("Zugbestand", f"{r['fleet']}", delta=f"{r['fleet'] - r['base_fleet']:+d} Zusatzzüge", delta_color="off")
                ref = {0: "phase", 1: "split"}.get(mode)
                if ref:
                    ls = stages[ref]["avg"]
                    st.info(f"Zum Vergleich die Lokalsuche dieser Demo auf demselben Netz: {ls:.1f} min ({STAGE_LABELS[ref]}). "
                            f"Unterschiede unter der Meldungsschwelle von {noise['split']:.1f} min sind kein Befund.")
                else:
                    st.info("Den Haltezeit-Spielraum kennt nur das exakte Modell; die Lokalsuche dieser Demo rechnet ihn nicht - das heißt nicht, dass nur CP-SAT ihn könnte.")
        elif ex:
            st.caption("Die Einstellungen haben sich seit dem letzten Lauf geändert - bitte erneut lösen.")
        cps, dws = R.cp_summary(res), R.dwell_summary(res)
        st.markdown("**Vorgerechnet (Messreihe, CP-SAT 20 s):**")
        st.dataframe(pd.DataFrame([
            {"Frage": f"Aufteilung (gleiche Züge), {cps['n']} Netze mit 6 Linien", "Ergebnis": f"CP-SAT Fund Ø {cps['cp_mean']:.1f} min, Schranke Ø {cps['bound_mean']:.1f}, bewiesen {cps['proven']}/{cps['n']}; "
             f"Lokalsuche mit {res['meta']['search_restarts']} Neustarts Ø {cps['ls40_mean']:.1f} (schlechter in {cps['ls40_worse']}, größte Abweichung {cps['ls40_max_dev']:.2f} min), mit 20 Neustarts Ø {cps['ls20_mean']:.1f} (schlechter in {cps['ls20_worse']}, größte Abweichung {cps['ls20_max_dev']:.2f})"},
            {"Frage": "Nur Phasen", "Ergebnis": f"bewiesen {cps['phase_proven']}/{cps['n']}; Lokalsuche = CP-SAT-Fund in {cps['phase_equal']}/{cps['n']}"},
            {"Frage": f"Haltezeit 1-3 min, {dws['n']} Netze", "Ergebnis": f"Aufteilung {dws['split0']:.1f} → mit Haltezeit {dws['dwell0']:.1f} min; mit Zusatzzug erlaubt {dws['dwell1']:.1f} min (Lokalsuche ohne Haltezeit mit Zusatzzug {dws['ls_add']:.1f}); "
             f"bewiesen {dws['proven'][0]} / {dws['proven'][1]} / {dws['proven'][2]} von {dws['n']}, Schranken Ø {dws['bound0']:.1f} / {dws['bound1']:.1f}"},
        ]), width="stretch", hide_index=True)

    with tabs[4]:
        st.caption("Alle Netzgrößen: Mittel ± Standardfehler je Stufe (Minuten).")
        st.dataframe(pd.DataFrame([{"Linien": k, **{STAGE_LABELS[s]: f"{R.cascade(res, k)[s]['mean']:.1f} ± {R.cascade(res, k)[s]['se']:.1f}" for s in STAGE_ORDER},
                                   "Untergrenze": f"{R.cascade(res, k)['lower_bound']['mean']:.1f}"} for k in C.LINES_OPTIONS]), width="stretch", hide_index=True)
        d_all = R.paired(res)
        st.dataframe(pd.DataFrame([{"Gepaarte Differenz": name, "Ø ± SE": f"{d['mean']:.2f} ± {d['se']:.2f}", "Median": f"{d['median']:.2f}", "P10": f"{d['p10']:.2f}", "P90": f"{d['p90']:.2f}"}
                                   for name, d in (("Gewinn der Aufteilung", d_all["split_gain"]), ("Gewinn der Zusatzzüge", d_all["add_gain"]),
                                                   ("Mehrkosten der Zugfolge (4 min)", d_all["headway_cost"]), ("Greedy minus Phasen-Optimum", d_all["greedy_gap"]))]),
                     width="stretch", hide_index=True)
        rc = res["recip"]
        st.caption(f"Kopplung geprüft: {rc['pairs']} Hin/Rück-Paare aus {rc['instances']} Netzen, Vollaufzählung des Versatzes: {rc['violations']} Verletzungen der Untergrenze, "
                   f"{rc['two_values_wrong']} Abweichungen von der Zwei-Werte-Regel; der kleinere Wert gilt für {100 * rc['low_share']:.1f} % der Versätze, c im Mittel {rc['mean_c']:.1f} min. "
                   f"Rauschen der Lokalsuche (5 Läufe je Netz, {res['noise']['split']['n']} Netze): Spannweite Median {res['noise']['split']['median']:.2f}, P95 {noise['split']:.2f} min (Aufteilung) "
                   f"bzw. {noise['add']:.2f} min (Zusatzzug) - daraus die Meldungsschwelle.")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        f"""
**Netz.** {C.N_STOPS} Haltestellen auf einem 100 × 100-Raster, symmetrische Fahrgast-Nachfrage mit zwei Hubs, 3 bis 10 Linien aus der Nachfrage-Greedy-Konstruktion der
transit-demo (Linien sind einfache Pfade, in beide Richtungen befahren). Alles ganzzahlig und mit dem Zufallsgenerator SplitMix64 erzeugt, damit jedes Netz überall dasselbe ist.

**Takt und Zeiten.** Jede Linie fährt in jede Richtung ein Mal je {C.T} Minuten. Fahrzeit je Abschnitt = Entfernung geteilt durch 5 (mindestens 2 min), Halt {C.DWELL} min, Standzeit am
Endpunkt mindestens {C.TURN_MIN} min. Der Zyklus (Hin + Wende + Rück + Wende) muss ein Vielfaches des Taktes werden; der nötige **Schlupf** geht in die Standzeit an den Endpunkten.

**Umsteiger.** Je Haltestellenpaar ohne Direktlinie die Route mit kürzester Fahrzeit über genau einen Umstieg; die Nachfrage steigt je zur Hälfte in jede Richtung um. Umsteigezeit =
Mindestzeit {C.TRANSFER_MIN} min + Wartezeit (mod {C.T}). Nur ein kleiner Teil der Fahrgäste steigt überhaupt um ({100 * min(r['transfer_share'] for r in R.nets(res)):.0f} bis
{100 * max(r['transfer_share'] for r in R.nets(res)):.0f} % der Nachfrage), der Rest fährt direkt; der Takt wirkt nur auf die Umsteiger.

**Die Stufen.** *Zur vollen Stunde*: alle Linien starten in Minute 0. *Zufällige Phasen*: Mittel über {C.RANDOM_SAMPLES} Zufallstakte. *Greedy*: Linien nach Umsteigegewicht einplanen, jede
bekommt den besten Versatz zu den schon eingeplanten. *Phasen optimiert*: Lokalsuche über die Versätze ({C.PHASE_RESTARTS} Neustarts). *+ Aufteilung*: zusätzlich darf die Standzeit
frei zwischen den beiden Endpunkten jeder Linie aufgeteilt werden ({C.SEARCH_RESTARTS} Neustarts), bei gleicher Zugzahl. *+ Zusatzzug*: die Linien mit Erlaubnis dürfen einen Zug mehr
einsetzen, dann darf die Standzeit um einen ganzen Takt länger werden.

**Warum Hin und zurück gekoppelt sind.** Bei festen Fahrzeiten legen die Fahrzeiten selbst fest, wie sich die beiden Wartezeiten eines Anschlusses zusammensetzen: schiebt man eine Linie,
wird die eine Wartezeit um so viel kürzer, wie die andere länger wird. Die Summe ist bis auf ganze Takte festgelegt. Die **Aufteilung der Standzeit** verändert die Phase der
Gegenrichtung und bricht diese Kopplung - deshalb bringt sie so viel. Das Verfahren wurde erstmals im Berliner U-Bahn-Fahrplan 2005 im Regelbetrieb eingesetzt (Liebchen 2008).

**Live-Netz und Messreihe.** Ein einzelnes Netz streut (die Lokalsuche ist eine Heuristik, zwei Läufe weichen bis etwa 1 min ab); die vorgerechnete Messreihe über 150 Netze steht deshalb
gleichberechtigt daneben, und Meldungen über einzelne Netze nennen eine Schwelle aus dem gemessenen Rauschen.

**Grenzen.** Synthetisches Netz, nur Umstiege über genau einen Umstieg, Nachfrage nicht nach Tageszeit, Fahrzeiten proportional zur Entfernung, keine lastabhängige Haltezeit, keine Block-
oder Weichenlogik bei der Zugfolge (die eigentliche Trassenvergabe ist der Baustein [Trassenkonflikt](https://sebastianhanisch-streckenkonflikt-demo.streamlit.app/)). Die Haltezeit-Spielräume rechnet nur das exakte Modell (CP-SAT, meist nicht bewiesen).
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ereignisse und Aktivitäten (PESP).** Ereignisse $i$ (Ankunft oder Abfahrt an einer Haltestelle in einer Richtung) haben einen Zeitpunkt $\pi_i \in [0, T)$. Eine Aktivität
$a=(i,j)$ hat die Spannung $x_a = \pi_j - \pi_i + T\,p_a \in [\ell_a, u_a]$ mit ganzzahligem $p_a$. Fahrt: $\ell=u=$ Fahrzeit; Halt: $[2,2]$ (exakt: $[1,3]$); Standzeit am Endpunkt:
$[4,\,4+T-1]$; Umstieg: $[3,\,3+T-1]$; Zugfolge: $[h,\,T-h]$ für zwei Abfahrten auf demselben Abschnitt.

**Ziel.** $\min \sum_a w_a x_a$ über die Umstiegsaktivitäten (nachfragegewichtete mittlere Wartezeit inkl. Mindestzeit). Zyklusbedingung je Linie: $\sum x_a = T\cdot v$ mit der Zugzahl $v$.

**Hin/Rück-Kopplung.** Für zwei gegenläufige Umstiege $a, a'$ desselben Anschlusses gilt bei festen Fahrzeiten
$x_a + x_{a'} = 2\cdot 3 + c + k\,T$ mit $c = (D_1 + D_2) \bmod T$ und $k \in \{0,1\}$; $k = 0$ gilt für genau $c+1$ der $T$ Versätze. Dabei sind $D_1, D_2$ die
Differenzen der Fahrplanminuten bei Phase 0 abzüglich der Mindestzeit. Die Summe aller Untergrenzen (Paare mit gleichem Gewicht je Richtung) ist die **Untergrenze der Phasen-Aufgabe**; sie wurde
an allen geprüften Paaren per Vollaufzählung des Versatzes bestätigt.

**Aufteilung.** Verschiebt man die Standzeit $s$ vom einen zum anderen Endpunkt, ändert sich die Phase der Gegenrichtung um $s \pmod T$ und damit $D_1 + D_2$; mit Zusatzzug sind Standzeiten
bis $T-1$ erlaubt (Zyklus $+T$).

**Suche.** Koordinatenabstieg: Linie für Linie der beste $(\text{Phase}, \text{Aufteilung})$ bei festen anderen Linien, für alle $T \cdot n_s$ Kandidaten auf einmal (vektorisierte Kandidatenmatrix,
je Eintrag gegen die volle Neubewertung geprüft); die Zugfolge ist ein Strafterm. **Exakt:** dasselbe Modell mit allen Aktivitäten in CP-SAT.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zum Thema: [Schienenverkehr optimieren](https://sebastianhanisch.net/schienenverkehr-optimierung.html)."
)
