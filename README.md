# Taktfahrplan: Umsteigen im Takt (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-taktfahrplan-demo.streamlit.app/)**

Interaktive **Fall-Demo** zum **Taktfahrplan** (Periodic Event Scheduling Problem, PESP; Serafini & Ukovich 1989) im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Fortsetzung von
[transit-demo](https://github.com/sebastian-hanisch/transit-demo) (Liniennetz, dort ausdrücklich „ohne Taktfrequenz/Fahrplan“) und **erster Baustein der Reihe
Bahn/Schienenverkehr**. Anders als [raptor-demo](https://github.com/sebastian-hanisch/raptor-demo) (Verbindungsauskunft auf einem *gegebenen* Fahrplan) und
[job-shop-demo](https://github.com/sebastian-hanisch/job-shop-demo) (Reihenfolgen ohne Periodizität) werden hier die **Abfahrtsminuten der Linien gewählt**, mit
periodischen Differenzbedingungen modulo T.

Jede Linie fährt in beide Richtungen alle 60 Minuten; gewählt wird, in welcher Minute sie startet, wie die Standzeit an den Endpunkten aufgeteilt wird, ob Linien einen
Zusatzzug bekommen und ob zwischen Zügen auf demselben Abschnitt eine Mindest-Zugfolge liegen muss. Gemessen wird die mittlere Wartezeit der Umsteiger.

Weitere Bausteine der Reihe: [Trassenkonflikt](https://github.com/sebastian-hanisch/trassenkonflikt-demo), [Fahrzeitreserve](https://github.com/sebastian-hanisch/fahrzeitreserve-demo), [Energieoptimale Fahrweise](https://github.com/sebastian-hanisch/energiefahrweise-demo), [Crew Pairing](https://github.com/sebastian-hanisch/crew-pairing-demo), [Ablaufberg](https://github.com/sebastian-hanisch/ablaufberg-demo). Die ganze Reihe mit Querverweisen auf verwandte Modelle steht auf der Seite [Schienenverkehr optimieren](https://sebastianhanisch.net/schienenverkehr-optimierung.html).

## Kernfrage

Wie lange wartet, wer im Taktnetz umsteigt, und welcher Spielraum im Fahrplan bringt wie viel und kostet was? Die Antwort ist nicht „die Heuristik schlägt das exakte
Verfahren“: Greedy und Lokalsuche sind für die Phasen-Aufgabe fast schon optimal. Der Befund ist eine **Kopplung von Hin- und Rückrichtung**.

## Befunde und Korrekturen gegenüber dem Plan

- **Hypothese „Vernetzung macht den Takt schwer“ ist widerlegt.** Der Kreisrang des Umsteigegraphen erklärt die Wartezeit kaum (r = −0.12 über 150 Netze); das
  Phasen-Optimum liegt bei 3 bis 10 Linien zwischen 23.4 und 24.3 min.
- **Die Messreihe wurde auf dem eigenen ganzzahligen Netzgenerator wiederholt.** Die erste Vorab-Reihe lief auf dem numpy-Generator der transit-demo (90 Netze); numpy
  garantiert keine versionsstabilen Zufallsströme. Die Befunde blieben in Richtung und Größe stehen (Aufteilungs-Gewinn 5.4 → 5.2 min, Anteil der Kopplung 72 → 73 %).
- **Zwei Befunde änderten den Bau.** Die Lokalsuche braucht 40 statt 20 Neustarts, um den CP-SAT-Fund zu erreichen. Die Kurve der Zusatzzüge ist mit sechs Neustarts je
  Schritt der größte Rechenposten; live läuft sie mit einem.
- **Korrektur im Plan:** die kürzeste Standzeit an einem Endpunkt ist 4 min (der Plan sagte zunächst 2); der Port der Demo wurde von 8951 auf 8955 verlegt, weil eine
  parallele Arbeit 8951 belegt hatte.

## Modell

- **Netz:** 20 Haltestellen auf einem 100 × 100-Raster, symmetrische ganzzahlige Nachfrage mit zwei Hubs, 3 bis 10 Linien aus dem Port der Nachfrage-Greedy-Konstruktion der
  transit-demo (einfache Pfade, höchstens 6 Halte, in beide Richtungen befahren). Alles ganzzahlig mit **SplitMix64** erzeugt (`tkt_rng.py`, `tkt_network.py`).
- **Zeiten:** Fahrzeit je Abschnitt = Entfernung (ganzzahlig über `isqrt`) / 5, mindestens 2 min; Halt 2 min; Standzeit am Endpunkt mindestens 4 min. Der Zyklus (Hin + Wende +
  Rück + Wende) muss ein Vielfaches des Taktes werden; der nötige **Schlupf** geht in die Standzeit, höchstens 59 min je Ende.
- **Umsteiger:** je Haltestellenpaar ohne Direktlinie die Route mit kürzester Fahrzeit über genau einen Umstieg, Nachfrage je zur Hälfte in jede Richtung. Umsteigezeit =
  Mindestzeit 3 min + Wartezeit (mod 60). Nur ein kleiner Teil der Nachfrage steigt um (6 bis 21 % je Netz); der Takt wirkt nur auf ihn.
- **PESP-Form:** π_j − π_i + T·p = x mit x ∈ [l, u]; Fahrt fest, Halt 2 min (exakt 1–3), Standzeit [4, 63], Umstieg [3, 62], Zugfolge [h, T−h].
- **Hin/Rück-Kopplung:** bei festen Fahrzeiten ist die Summe der beiden Wartezeiten eines Anschlusses 2·3 + c oder genau einen Takt mehr, mit c = (D1 + D2) mod T; der kleinere
  Wert gilt für genau c + 1 der 60 Versätze. Die Summe aller dieser Untergrenzen begrenzt die Phasen-Aufgabe von unten; die Aufteilung der Standzeit bricht sie.

## Methodik

- **Stufen** (`tkt_evaluation.py`): zur vollen Stunde, zufällige Phasen (Mittel über 300 Ziehungen), Greedy nach Umsteigegewicht, Phasen optimiert (20 Neustarts),
  + Aufteilung der Standzeit (40 Neustarts, gleiche Zugzahl), + Zusatzzug (Erlaubnis für k Linien, Kurve mit Warmstart).
- **Suche** (`tkt_search.py`): Koordinatenabstieg Linie für Linie über alle (Phase, Aufteilung)-Kandidaten gleichzeitig (vektorisierte Kandidatenmatrix, jeder Eintrag gegen die volle
  Neubewertung geprüft); die Mindest-Zugfolge ist ein Strafterm. Zufall nur über einen übergebenen SplitMix64.
- **Exakt** (`tkt_exact.py`): dasselbe Modell mit allen Aktivitäten in OR-Tools CP-SAT, wahlweise mit Haltezeit-Spielraum und Zusatzzug; Fund, Schranke, Status. Eigener Tab, eigener
  Knopf.
- **Vorgerechnete Messreihe** (`tools/sweep.py`, `tools/dump_sweep.py` → `data/tkt_results.json`): 30 Netze je Linienzahl (3, 5, 6, 8, 10), Zugfolge 3 / 4 / 5 min, CP-SAT-Vergleich
  auf 12 Netzen, Haltezeit-Reihe auf 15 Netzen, Rauschen der Lokalsuche (5 Läufe je Netz, 30 Netze), Prüfung der Kopplung an echten Paaren. Live läuft nur das gewählte Netz.
- **Meldungsschwelle:** Aussagen über ein einzelnes Netz („die Aufteilung bringt …“) nennen eine Schwelle aus dem gemessenen Rauschen (P95 der Spannweite zweier Läufe); darunter
  steht „im Rauschen“, nie ein Befund.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Mittel ± Standardfehler über **150 Netze** (je 30 bei 3, 5, 6, 8 und 10 Linien), 20 Haltestellen, Takt 60 min.

| Frage | Befund |
|---|---|
| Wie lange wartet, wer umsteigt (6 Linien, 30 Netze)? | Zufällige Phasen **32.5 min**, optimierte Phasen **24.1 min**, mit freier Aufteilung der Standzeit **19.4 min** (gleiche Zugzahl), mit Zusatzzügen 16.3 min; ideal wären 3 min. |
| Wie viele warten höchstens 10 min? | Bei optimierten Phasen **28 %**, mit Aufteilung **46 %**, mit Zusatzzügen 62 %. Mit Zusatzzügen warten bei 6 Linien noch **38 %** länger als 10 min. |
| Warum geht es nicht besser? | Die Hin/Rück-Kopplung erzwingt im Mittel **73 %** des Aufschlags über die Mindestzeit (je Netzgröße 70–74 %). An 398 echten Hin/Rück-Paaren per Vollaufzählung: 0 Verletzungen der Untergrenze, 0 Abweichungen von der Zwei-Werte-Regel; der kleinere Wert gilt für 50.8 % der Versätze, c im Mittel 29.5 min. |
| Macht ein dichteres Netz den Takt schwerer? | **Nein.** Phasen-Optimum 24.0 / 24.3 / 24.1 / 23.4 / 23.5 min bei 3 / 5 / 6 / 8 / 10 Linien; Kreisrang gegen Phasen-Optimum r = −0.12. |
| Was bringt die Aufteilung der Standzeit? | **5.2 ± 0.2 min** weniger Wartezeit als der beste Takt mit Standard-Standzeit (Median 4.8, P10 2.2, P90 8.3), in allen 150 Netzen positiv (150/150, kleinster Gewinn 1.02 min), bei gleicher Zugzahl. |
| Und Zusatzzüge? | Gewinn gegenüber der Aufteilung 3.4 ± 0.2 min (alle Linien dürfen); abnehmender Ertrag: bei 6 Linien 19.4 → 17.8 → 17.0 → 16.5 → 16.4 min bei Erlaubnis für 0 bis 4 Linien, danach unter 0.2 min je Schritt. |
| Was kostet eine Mindest-Zugfolge von 4 min? | **1.1 ± 0.1 min** Wartezeit im Mittel über alle Netze (Median 0.5); bei 3 / 5 / 6 / 8 / 10 Linien 0.5 / 0.9 / 0.8 / 1.3 / 2.0 min. Überall ein zulässiger Fahrplan gefunden. Ohne die Bedingung verletzt die Aufteilungs-Lösung sie bei 10 Linien in 30 von 30 Netzen. |
| Ist Greedy schon gut? | Ja: Greedy liegt 0.6 ± 0.1 min (2.7 %) über dem Phasen-Optimum, in 46 von 150 Netzen genau gleich. Die Phasen-Aufgabe selbst ist leicht. |
| Schlägt CP-SAT die Lokalsuche? | Nein. Aufteilung, 12 Netze mit 6 Linien: CP-SAT-Fund Ø 19.7 min, Schranke Ø 14.8, **bewiesen nur 1/12**; Lokalsuche mit 40 Neustarts Ø 19.7 (schlechter in 3 Netzen, größte Abweichung 0.08 min), mit 20 Neustarts schlechter in 7 Netzen (bis 0.64 min). Nur Phasen: bewiesen in 6 von 12, Lokalsuche gleich dem Fund in 11. |
| Haltezeit-Spielraum (nur CP-SAT)? | 15 Netze, 20 s: Aufteilung 19.7 → mit Haltezeit 1–3 min 16.2 min; mit Zusatzzug 10.4 min. Bewiesen nur 2 / 1 / 5 von 15, Schranken Ø 9.0 / 6.8 min: Funde, keine Optima. |
| Wie stark rauscht die Lokalsuche je Netz? | Spannweite zweier Läufe: Aufteilung Median 0.03, **P95 0.73 min**, Maximum 1.82; Zusatzzug P95 0.49 min. Daraus die Meldungsschwelle. |

## Ehrliche Grenzen

- Synthetisches Netz mit wenigen Umsteigern; nur Umstiege über genau einen Umstieg; Nachfrage nicht nach Tageszeit; Fahrzeiten proportional zur Entfernung; keine lastabhängige
  Haltezeit. Die Zahlen belegen Richtung und Größenordnung auf diesen Netzen, keine absoluten Minutenwerte realer Netze.
- Die **Zugfolge** ist nur ein Vorgriff: Abstand der Abfahrten auf demselben gerichteten Abschnitt, keine Block-, Weichen- oder Überholungslogik. Die eigentliche Trassenvergabe ist der Baustein
  [Trassenkonflikt](https://github.com/sebastian-hanisch/trassenkonflikt-demo).
- Der **Zusatzzug** zählt nur den Bestand (Zyklusdauer geteilt durch den Takt); ein Fahrzeugumlauf im Detail (Depot, Wartung) ist nicht modelliert.
- Die Lokalsuche ist eine Heuristik: ein einzelnes Netz streut bis etwa 1 min. Deshalb stehen Messreihe und Live-Netz gleichberechtigt nebeneinander.
- Den **Haltezeit-Spielraum** rechnet nur das exakte Modell, überwiegend unbewiesen. Das heißt nicht, dass nur CP-SAT ihn könnte; die Lokalsuche dafür ist nicht gebaut.
- Die Live-Kurve der Zusatzzüge nutzt einen Neustart je Schritt (schnell, etwas gröber); die Messreihe rechnet mit sechs.

## Tests

Siehe `tests/`: Fahrplanmodell und Suche an Mini-Instanzen von Hand gerechnet (CP-SAT = Vollaufzählung, Delta-Prüfung der Kandidatenmatrix, Hin/Rück-Summe), SplitMix64 gegen
Referenzwerte und der Linienport gegen eingefrorene Referenzlinien des Originals, Zweig-Tests (jede Stufe weicht auf mindestens einem Netz von der Nachbarstufe ab), Statistik an einer
von Hand gerechneten Mini-Ergebnisdatei, Preset-Kriterien mit künstlichen Werten (jedes Kriterium kippt einzeln), `test_claims.py` (jede README-Zahl gegen die Ergebnisdatei),
AppTest-Rauchtests (Voreinstellung, jedes Preset, Randwerte, Permalink, Exakt-Tab). Keine Wall-Clock-Assertions, keine exakten Werte der Heuristik.

```
python -m pytest tests -q
```

`tools/mutation_check.py` baut einzelne Fehler in die Module ein und prüft, ob die Tests sie finden.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `tkt_constants.py` | feste Annahmen, Regler-Grenzen, Presets |
| `tkt_rng.py`, `tkt_network.py` | SplitMix64; ganzzahliger Netz- und Liniengenerator |
| `tkt_timetable.py` | Linien, Ereigniszeiten, Umsteiger, Wartezeiten, Hin/Rück-Untergrenze |
| `tkt_search.py` | Greedy, Phasen-Abstieg, Aufteilung, Zusatzzug, Zugfolge, Kurve |
| `tkt_exact.py` | CP-SAT-Modell |
| `tkt_evaluation.py` | Stufen für ein Netz, Meldungen, Kopplungs-Abschnitt |
| `tkt_results.py` | Auswertung der vorgerechneten Messreihe |
| `tkt_stories.py`, `tkt_presets.py` | Preset-Kriterien; Permalink und Presets |
| `tkt_visualization.py`, `tkt_pdf_export.py` | Plotly-Figuren; Fahrplanaushang als PDF |
| `data/tkt_results.json` | Ergebnisse der Messreihe |
| `tools/` | `sweep.py`, `dump_sweep.py`, `preset_search.py`, `mutation_check.py`, `PRESET_SWEEP.md` |

## Bewusst nicht umgesetzt

Fahrzeugumlauf im Detail (bei einem Depot ein Zuordnungs- bzw. Min-Cost-Flow-Problem, kein eigener Baustein) und Störungsmanagement; die Trassenvergabe mehrerer Bahnunternehmen ist der Baustein [Trassenkonflikt](https://github.com/sebastian-hanisch/trassenkonflikt-demo), der Personaleinsatz das [Crew Pairing](https://github.com/sebastian-hanisch/crew-pairing-demo), die Zugbildung der [Ablaufberg](https://github.com/sebastian-hanisch/ablaufberg-demo). Außerdem: Takt T = 30,
Mehrfach-Umstiege, Nachfrage nach Tageszeit, Haltezeit-Spielraum in der Lokalsuche, Kalibrierung an realen Fahrplandaten.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Die Messreihe neu erzeugen (Minuten bis Stunden): `python tools/sweep.py nets 6 tools/raw/nets_6.jsonl` (und die übrigen Unterbefehle, siehe Kopf von `tools/sweep.py`), dann `python tools/dump_sweep.py`.

Gebaut mit Streamlit, Plotly, NumPy, OR-Tools und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zum Thema: [Schienenverkehr optimieren](https://sebastianhanisch.net/schienenverkehr-optimierung.html).
