"""Fahrplanaushang als PDF (fpdf2, Kernschrift Helvetica: Umlaute gehen, Gedankenstrich, Euro-Zeichen, Emoji und U+2212 nicht)."""
from __future__ import annotations

from fpdf import FPDF

from tkt_evaluation import view_schedule

_REPLACE = {"–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "→": "->", "↔": "<->", "…": "...", "€": "EUR", "·": "."}


def _clean(text: str) -> str:
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_timetable_pdf(inst, stage: dict, title: str, summary_lines: list) -> bytes:
    """Taktfahrplan: je Linie die Abfahrtsminute (im Takt T) an jedem Halt, Hin- und Rückrichtung; vorweg Kennzahlen."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in summary_lines:
        pdf.multi_cell(0, 5, _clean(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.5, _clean(f"Alle Zeiten sind Abfahrtsminuten im {inst.T}-Minuten-Takt: der Zug fährt jede Stunde zu dieser Minute ab."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for li, rows in view_schedule(inst, stage):
        stops = " - ".join(str(s) for s, _, _ in rows)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _clean(f"Linie {li + 1}: Halte {stops}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(30, 6, "Haltestelle", border=1)
        pdf.cell(50, 6, "Hinrichtung (Minute)", border=1)
        pdf.cell(50, 6, "Rückrichtung (Minute)", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for s, hin, rueck in rows:
            pdf.cell(30, 6, str(s), border=1)
            pdf.cell(50, 6, "Endhalt" if hin is None else f"{hin:02d}", border=1)
            pdf.cell(50, 6, "Endhalt" if rueck is None else f"{rueck:02d}", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    return bytes(pdf.output())
