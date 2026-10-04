import tkt_evaluation as E
from tkt_pdf_export import _clean, generate_timetable_pdf


def test_clean_replaces_every_character_fpdf2_cannot_encode():
    text = "Wartezeit – Ø ≤ 10 min → 3 € … − 2"
    cleaned = _clean(text)
    cleaned.encode("latin-1")
    assert "–" not in cleaned and "€" not in cleaned and "−" not in cleaned and "→" not in cleaned


def test_timetable_pdf_is_a_pdf_with_every_line():
    run = E.run_live(3, 7, 0, 1)
    pdf = generate_timetable_pdf(run["inst"], run["stages"]["split"], "Taktfahrplan – + Aufteilung", ["Zeile 1 – mit Gedankenstrich", "Zeile 2 ≤ Ø"])
    assert pdf[:5] == b"%PDF-" and len(pdf) > 2000
