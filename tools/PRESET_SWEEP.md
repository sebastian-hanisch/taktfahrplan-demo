# Preset-Abstimmung (tools/preset_search.py)

Seeds 500..539 je Preset, außerhalb der Messreihen-Seeds (3: 100-129, 5: 200-229, 6: 0-29, 8: 300-329, 10: 400-429). Ein Seed besteht, wenn alle Kriterien des Presets gelten (siehe `tkt_stories.py`; gleiche Kriterien prüft `tests/test_claims.py` an den gewählten Seeds).

## Standard

Kriterien: split_over_noise (Aufteilungs-Gewinn des Netzes über der Meldungsschwelle); le10_grows (Anteil Umsteiger ≤ 10 min wächst von Phasen zu Aufteilung um mindestens 5 Prozentpunkte); greedy_close (Messreihe: Greedy höchstens 5 % über dem Phasen-Optimum)

Bestanden: 36 von 40 Seeds.

Gewählt: Seed 501 (tatsächlicher Seed 501), Aufteilungs-Gewinn 4.83 min, Zusatzzug-Gewinn 3.80 min, Zugfolge-Mehrkosten 0.00 min.

## Dichtes Netz

Kriterien: ten_lines (das Netz hat wirklich 10 Linien); density_not_the_problem (Messreihe: Phasen-Optimum bei 10 Linien höchstens 1,5 min über dem bei 3 Linien); headway_grows_with_density (Messreihe: Zugfolge-Mehrkosten bei 10 Linien größer als bei 3 Linien)

Bestanden: 40 von 40 Seeds.

Gewählt: Seed 528 (tatsächlicher Seed 528), Aufteilungs-Gewinn 4.72 min, Zusatzzug-Gewinn 1.23 min, Zugfolge-Mehrkosten 0.00 min.

## Kleines Netz

Kriterien: three_lines (das Netz hat wirklich 3 Linien); low_rank (Kreisrang des Umsteigegraphen höchstens 2); coupling_dominates (Messreihe: Untergrenze erzwingt mindestens 70 % des Aufschlags über die Mindestzeit); extra_train_over_noise (ein Zusatzzug bringt mehr als die Meldungsschwelle des Zusatzzugs)

Bestanden: 22 von 40 Seeds.

Gewählt: Seed 535 (tatsächlicher Seed 535), Aufteilungs-Gewinn 8.17 min, Zusatzzug-Gewinn 3.99 min, Zugfolge-Mehrkosten 0.00 min.

## Enge Zugfolge

Kriterien: headway_over_noise (Mehrkosten der Zugfolge auf dem Netz über der Meldungsschwelle); optimized_stages_feasible (alle optimierten Stufen halten die Zugfolge ein); headway_significant (Messreihe: Mehrkosten bei 8 Linien und 5 min über 2 Standardfehlern)

Bestanden: 33 von 40 Seeds.

Gewählt: Seed 510 (tatsächlicher Seed 510), Aufteilungs-Gewinn 6.58 min, Zusatzzug-Gewinn 1.56 min, Zugfolge-Mehrkosten 1.84 min.

## Viele Zusatzzüge

Kriterien: saturation (Gewinn des letzten Schritts der Kurve höchstens 0,2 min); trains_used (Zusatzzüge genutzt: mindestens einer, höchstens 4); extra_train_over_noise (alle Zusatzzüge zusammen bringen mehr als die Meldungsschwelle); sweep_saturates (Messreihe: Gewinn des letzten Kurvenschritts bei 6 Linien höchstens 0,2 min)

Bestanden: 35 von 40 Seeds.

Gewählt: Seed 517 (tatsächlicher Seed 517), Aufteilungs-Gewinn 8.10 min, Zusatzzug-Gewinn 3.19 min, Zugfolge-Mehrkosten 0.00 min.

