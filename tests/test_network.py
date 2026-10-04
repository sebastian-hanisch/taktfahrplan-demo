"""Tests für SplitMix64 und den ganzzahligen Netzgenerator."""
import numpy as np

import tkt_network as N
import tkt_rng


def test_splitmix64_reference_values():
    # Referenzwerte des Original-Algorithmus (Vigna) für Seed 0
    r = tkt_rng.SplitMix64(0)
    assert r.next() == 0xE220A8397B1DCDAF
    assert r.next() == 0x6E789E6AA1B965F4
    assert r.next() == 0x06C45D188009454F
    r = tkt_rng.SplitMix64(12345)
    vals = [r.below(7) for _ in range(2000)]
    assert set(vals) == set(range(7))                     # alle Werte kommen vor, nie außerhalb
    assert tkt_rng.SplitMix64(5).randrange(1000) == tkt_rng.SplitMix64(5).below(1000)


def test_generator_properties_and_determinism():
    c1, d1, h1 = N.generate_network(20, 7)
    c2, d2, h2 = N.generate_network(20, 7)
    assert (c1 == c2).all() and (d1 == d2).all() and h1 == h2
    assert c1.min() >= 5 and c1.max() <= 95 and c1.dtype == np.int64
    assert (d1 == d1.T).all() and (np.diag(d1) == 0).all() and d1.min() >= 0
    assert len(set(h1)) == len(h1) == 2
    # Hubs ziehen Nachfrage: Zeilensumme der Hubs über dem Median der übrigen (an 20 Seeds fast immer)
    hits = 0
    for seed in range(20):
        _, d, hubs = N.generate_network(20, seed)
        rest = [int(d[i].sum()) for i in range(20) if i not in hubs]
        hits += all(int(d[h].sum()) > np.median(rest) for h in hubs)
    assert hits >= 18, hits
    # hub_pct = 0: kein Hub-Aufschlag, Nachfrage gleich dem Basiswert 1..10
    _, d0, _ = N.generate_network(20, 3, hub_pct=0)
    assert d0.min() >= 0 and d0.max() <= 10


# Referenzlinien des ORIGINALS (transit-demo, demand_greedy_construction) auf derselben ganzzahligen Eingabe; in AP 0 einmalig mit dem
# Original erzeugt (Port = Original für 648 Parameterkombinationen geprüft) und hier eingefroren.
FROZEN = {
    (14, 6, 6, 0): [[10, 11, 2, 12, 8, 5], [0, 11, 7, 10, 9, 1], [10, 6, 11, 2, 4, 3], [10, 13, 11, 2, 8, 12], [8, 2, 11, 0, 6, 10], [10, 1, 6, 11, 2, 12]],
    (20, 3, 6, 100): [[9, 2, 8, 1, 7, 11], [13, 14, 2, 12, 19, 18], [16, 5, 6, 8, 2, 3]],
    (20, 6, 6, 5): [[4, 2, 8, 11, 15, 5], [1, 16, 17, 0, 11, 14], [13, 11, 8, 18, 19, 9], [2, 12, 11, 8, 10, 15], [3, 15, 7, 8, 11, 2], [15, 11, 6, 10, 8, 2]],
    (20, 10, 6, 400): [[7, 11, 15, 19, 9, 2], [10, 19, 15, 18, 13, 8], [0, 19, 15, 3, 16, 5], [9, 1, 15, 19, 17, 14], [9, 12, 4, 19, 15, 6], [11, 15, 1, 6, 19, 9], [11, 19, 15, 1, 16, 9], [2, 9, 4, 11, 19, 15], [11, 15, 19, 4, 16, 9], [11, 15, 19, 5, 10, 9]],
}


def test_port_equals_frozen_reference():
    assert len(FROZEN) == 4 and all(ref for ref in FROZEN.values()), "Referenzen müssen gefüllt sein (kein stiller Leertest)"
    for key, ref in FROZEN.items():
        n_stops, n_lines, max_len, seed = key
        coords, demand, _ = N.generate_network(n_stops, seed)
        assert N.demand_greedy(coords, demand, n_lines, max_len) == ref, key


def test_built_instances_are_usable():
    for n_stops, n_lines in ((14, 6), (20, 3), (20, 5), (20, 6), (20, 8), (20, 10)):
        for seed in range(30):
            inst, demand, coords = N.build_instance_int(n_stops, n_lines, 6, seed)
            assert len(inst.lines) >= 2 and inst.transfers, (n_stops, n_lines, seed)
            for line in inst.lines:
                assert len(set(line.stops)) == len(line.stops) and all(r >= 2 for r in line.run)
                assert len(line.run) == len(line.stops) - 1
