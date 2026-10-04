"""Plotly-Figuren der Taktfahrplan-Demo. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Chart hängen bleibt;
Karte mit scaleanchor nur mit autorange und Eckmarkern (Plotly friert sonst den Bereich beim ersten Zeichnen ein)."""
from __future__ import annotations


import plotly.graph_objects as go

import tkt_constants as C

LABELS = {"same": "zur vollen\nStunde", "random": "zufällige\nPhasen", "greedy": "Greedy", "phase": "Phasen\noptimiert",
          "split": "+ Aufteilung\nWende", "add": "+ Zusatzzug\nerlaubt"}
STAGE_ORDER = ("same", "random", "greedy", "phase", "split", "add")


def _lock(fig, height=360, **layout):
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=60), font=dict(size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_staircase(stages: dict, lower_bound: float, reference: dict | None = None, ref_label: str = "Messreihe Ø") -> go.Figure:
    """Mittlere Wartezeit je Stufe für dieses Netz; optional als Punkte mit Standardfehler die Messreihe (Ø über Netze gleicher Größe)."""
    xs = [LABELS[k].replace("\n", " ") for k in STAGE_ORDER]
    ys = [stages[k]["avg"] for k in STAGE_ORDER]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[C.STAGE_COLORS[k] for k in STAGE_ORDER], name="dieses Netz",
                           text=[f"{v:.1f}" for v in ys], textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=13),
                           hovertemplate="%{x}: %{y:.1f} min<extra></extra>"))
    if reference:
        fig.add_trace(go.Scatter(x=xs, y=[reference[k]["mean"] for k in STAGE_ORDER], mode="markers", name=ref_label,
                                 marker=dict(symbol="diamond", size=10, color="#1c2430"),
                                 error_y=dict(type="data", array=[reference[k]["se"] for k in STAGE_ORDER], visible=True, color="#1c2430"),
                                 hovertemplate="%{x}: Ø %{y:.1f} min<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[lower_bound] * len(xs), mode="lines", line=dict(color="#c0392b", dash="dash", width=2),
                             name=f"Untergrenze Hin/Rück-Kopplung (nur Phasen): {lower_bound:.1f}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[C.TRANSFER_MIN] * len(xs), mode="lines", line=dict(color="#5b6b80", dash="dot", width=2),
                             name="ideal: 3 min (nur die Mindest-Umsteigezeit)", hoverinfo="skip"))
    top = max(max(ys), lower_bound) * 1.1
    fig.update_yaxes(title_text="Ø Umsteige-Wartezeit (min)", range=[0, top])
    return _lock(fig, 380)


def build_network_map(inst, coords, stage_lines=None, highlight_stop: int | None = None, allowed_lines=None) -> go.Figure:
    """Netzkarte: Linien in Farbe (auf den Haltestellen, bei gemeinsamen Abschnitten gestaffelte Breite), Haltestellen mit Nummer, optional ein hervorgehobener Umsteigeknoten."""
    fig = go.Figure()
    n = len(inst.lines)
    for li, line in enumerate(inst.lines):
        # Linien liegen genau auf den Haltestellen; wo mehrere denselben Abschnitt befahren, liegt die breiteste (erste Linie) unten und
        # jede weitere schmaler darüber, damit alle Farben sichtbar bleiben und jede Kante an der Haltestelle endet.
        xs, ys = [], []
        for a, b in zip(line.stops, line.stops[1:]):
            xs += [float(coords[a][0]), float(coords[b][0]), None]
            ys += [float(coords[a][1]), float(coords[b][1]), None]
        extra = " (Zusatzzug erlaubt)" if allowed_lines and allowed_lines[li] else ""
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.LINE_COLORS[li % len(C.LINE_COLORS)], width=2.5 + 1.1 * (n - 1 - li)),
                                 opacity=0.9, name=f"Linie {li + 1}{extra}", hoverinfo="name"))
    used = sorted({s for line in inst.lines for s in line.stops})
    fig.add_trace(go.Scatter(x=[float(coords[s][0]) for s in used], y=[float(coords[s][1]) for s in used], mode="markers+text",
                             marker=dict(size=9, color="white", line=dict(color="#1c2430", width=1.5)), text=[str(s) for s in used],
                             textposition="top right", textfont=dict(size=10), name="Haltestellen", showlegend=False,
                             hovertemplate="Haltestelle %{text}<extra></extra>"))
    if highlight_stop is not None:
        fig.add_trace(go.Scatter(x=[float(coords[highlight_stop][0])], y=[float(coords[highlight_stop][1])], mode="markers",
                                 marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color="#c0392b", width=3)), name="gewählter Anschluss",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))
    fig.update_xaxes(visible=False, fixedrange=True, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, fixedrange=True, autorange="reversed")
    return fig


def build_coupling_chart(hin, rueck, sums, current_offset: int, low: int) -> go.Figure:
    """Wartezeiten beider Richtungen eines Anschlusses über alle Versätze der zweiten Linie, dazu ihre Summe."""
    xs = list(range(len(hin)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=hin, mode="lines", name="Wartezeit Hinrichtung", line=dict(color="#2a6fb0", width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=rueck, mode="lines", name="Wartezeit Rückrichtung", line=dict(color="#2e7d4f", width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=sums, mode="lines", name="Summe beider", line=dict(color="#c0392b", width=2.5, dash="dash")))
    fig.add_vline(x=current_offset, line_color="#1c2430", line_width=1.5)
    fig.add_trace(go.Scatter(x=[current_offset] * 3, y=[hin[current_offset], rueck[current_offset], sums[current_offset]], mode="markers",
                             marker=dict(size=9, color=["#2a6fb0", "#2e7d4f", "#c0392b"]), showlegend=False,
                             hovertemplate="Versatz %{x}: %{y} min<extra></extra>"))
    fig.update_xaxes(title_text="Versatz der zweiten Linie gegen die erste (Minuten im Takt)", range=[0, len(hin) - 1])
    fig.update_yaxes(title_text="Umsteige-Wartezeit (min)", range=[0, max(sums) * 1.08])
    return _lock(fig, 360)


def build_cascade_chart(cascade: dict, n_lines: int) -> go.Figure:
    """Messreihe: Mittel ± Standardfehler je Stufe (Netze gleicher Größe)."""
    xs = [LABELS[k].replace("\n", " ") for k in STAGE_ORDER]
    fig = go.Figure(go.Bar(x=xs, y=[cascade[k]["mean"] for k in STAGE_ORDER], marker_color=[C.STAGE_COLORS[k] for k in STAGE_ORDER],
                           error_y=dict(type="data", array=[cascade[k]["se"] for k in STAGE_ORDER], visible=True),
                           text=[f"{cascade[k]['mean']:.1f}" for k in STAGE_ORDER], textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=13),
                           hovertemplate="%{x}: Ø %{y:.1f} min<extra></extra>", name="Ø über die Netze"))
    fig.add_trace(go.Scatter(x=xs, y=[cascade["lower_bound"]["mean"]] * len(xs), mode="lines", line=dict(color="#c0392b", dash="dash", width=2),
                             name=f"Untergrenze Ø {cascade['lower_bound']['mean']:.1f}", hoverinfo="skip"))
    fig.update_yaxes(title_text="Ø Umsteige-Wartezeit (min)", range=[0, 45])
    return _lock(fig, 340)


def build_size_chart(curves: dict, marker_lines: int | None = None) -> go.Figure:
    """Wartezeit über die Netzgröße für Phasen, + Aufteilung, + Zusatzzug."""
    names = {"phase": "Phasen optimiert", "split": "+ Aufteilung", "add": "+ Zusatzzug erlaubt"}
    fig = go.Figure()
    for s, label in names.items():
        xs = sorted(curves[s])
        fig.add_trace(go.Scatter(x=[f"{x} Linien" for x in xs], y=[curves[s][x] for x in xs], mode="lines+markers+text", name=label,
                                 line=dict(color=C.STAGE_COLORS[s], width=2.5), text=[f"{curves[s][x]:.1f}" for x in xs], textposition="top center"))
    fig.update_yaxes(title_text="Ø Wartezeit (min)", range=[0, 32])
    return _lock(fig, 340)


def build_curve_chart(live_curve: list, base_fleet: int, ref_points: list | None = None, ref_label: str = "Messreihe Ø") -> go.Figure:
    """Wartezeit gegen tatsächlich genutzte Zusatzzüge: dieses Netz (live) und optional Messreihe."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[e["fleet"] - base_fleet for e in live_curve], y=[e["avg"] for e in live_curve], mode="lines+markers",
                             name="dieses Netz", line=dict(color=C.STAGE_COLORS["add"], width=2.5),
                             text=[f"Erlaubnis für {e['allowed']} Linien" for e in live_curve], hovertemplate="%{text}: %{x} Zusatzzüge, %{y:.1f} min<extra></extra>"))
    if ref_points:
        fig.add_trace(go.Scatter(x=[p[1] for p in ref_points], y=[p[2] for p in ref_points], mode="lines+markers", name=ref_label,
                                 line=dict(color="#1c2430", width=2, dash="dot"), marker=dict(symbol="diamond"),
                                 hovertemplate="Ø %{x:.1f} Zusatzzüge: %{y:.1f} min<extra></extra>"))
    fig.update_xaxes(title_text="tatsächlich genutzte Zusatzzüge")
    fig.update_yaxes(title_text="Ø Wartezeit (min)")
    return _lock(fig, 340)


def build_headway_chart(rows: list, h: int) -> go.Figure:
    """Mehrkosten der Mindest-Zugfolge h nach Netzgröße (Mittel ± Standardfehler)."""
    sel = [r for r in rows if r["h"] == h]
    fig = go.Figure(go.Bar(x=[f"{r['lines']} Linien" for r in sel], y=[r["cost"] for r in sel], error_y=dict(type="data", array=[r["se"] for r in sel], visible=True),
                           marker_color="#8e44ad", text=[f"{r['cost']:.1f}" for r in sel], textposition="outside",
                           hovertemplate="%{x}: +%{y:.2f} min<extra></extra>"))
    fig.update_yaxes(title_text=f"Mehrkosten bei {h} min Zugfolge (min)", rangemode="tozero")
    return _lock(fig, 320)
