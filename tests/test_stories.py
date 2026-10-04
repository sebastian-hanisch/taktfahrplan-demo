"""Preset-Kriterien mit künstlichen Werten: jedes Kriterium kippt einzeln an seiner Schwelle (DEMO-PLAYBOOK: Preset-Disziplin)."""
import pytest

import tkt_constants as C
import tkt_stories as S

LINES_OF = {"Standard": 6, "Dichtes Netz": 10, "Kleines Netz": 3, "Enge Zugfolge": 8, "Viele Zusatzzüge": 6}

GOOD = {"split_gain": 2.0, "thr_split": 0.95, "thr_add": 0.57, "le10_split": 0.5, "le10_phase": 0.3, "greedy_gap_pct": 2.0, "phase10_minus_phase3": -0.5,
        "hw_cost_10": 1.9, "hw_cost_3": 0.5, "rank": 1, "coupling_share_3": 0.74, "add_gain": 3.0, "headway_cost": 2.0, "violations_optimized": 0,
        "hw_cost_8_h5": 2.0, "hw_cost_8_h5_se": 0.4, "last_step_gain": 0.1, "extra_trains": 3, "sweep_last_step_gain": 0.05}

# Kennung -> (Schlüssel, Wert knapp auf der falschen Seite, Wert knapp auf der richtigen Seite)
FLIPS = {
    "split_over_noise": ("split_gain", 0.95, 0.96), "le10_grows": ("le10_split", 0.32, 0.5), "greedy_close": ("greedy_gap_pct", 5.1, 5.0),
    "ten_lines": ("n_lines", 9, 10), "density_not_the_problem": ("phase10_minus_phase3", 1.6, 1.5), "headway_grows_with_density": ("hw_cost_10", 0.5, 0.6),
    "three_lines": ("n_lines", 4, 3), "low_rank": ("rank", 3, 2), "coupling_dominates": ("coupling_share_3", 0.69, 0.70),
    "extra_train_over_noise": ("add_gain", 0.57, 0.58),
    "headway_over_noise": ("headway_cost", 0.95, 0.96), "optimized_stages_feasible": ("violations_optimized", 1, 0),
    "headway_significant": ("hw_cost_8_h5", 0.8, 0.9),
    "saturation": ("last_step_gain", 0.21, 0.2), "trains_used": ("extra_trains", 0, 1), "sweep_saturates": ("sweep_last_step_gain", 0.21, 0.2),
}


def facts_for_preset(name):
    return {**GOOD, "n_lines": LINES_OF[name]}


@pytest.mark.parametrize("name", C.PRESET_ORDER)
def test_good_facts_satisfy_every_criterion_of_the_preset(name):
    assert all(ok for _, _, ok in S.check(name, facts_for_preset(name)))
    assert LINES_OF[name] == C.PRESETS[name]["lines"]


@pytest.mark.parametrize("name", C.PRESET_ORDER)
def test_each_criterion_flips_alone_at_its_threshold(name):
    for cid, text, _ in S.check(name, facts_for_preset(name)):
        key, bad, good = FLIPS[cid]
        for value, expect in ((bad, False), (good, True)):
            facts = {**facts_for_preset(name), key: value}
            outcome = {c: ok for c, _, ok in S.check(name, facts)}
            assert outcome[cid] is expect, (name, cid, key, value)
            assert all(ok for c, ok in outcome.items() if c != cid), (name, cid, "ein anderes Kriterium kippte mit")


def test_trains_used_has_an_upper_bound_too():
    facts = {**facts_for_preset("Viele Zusatzzüge"), "extra_trains": 5}
    assert not dict((c, ok) for c, _, ok in S.check("Viele Zusatzzüge", facts))["trains_used"]


def test_every_criterion_is_covered_by_a_flip_and_has_a_readable_text():
    ids = {cid for crit in S.CRITERIA.values() for cid, _, _ in crit}
    assert ids == set(FLIPS)
    assert all(len(text) > 15 for crit in S.CRITERIA.values() for _, text, _ in crit)
