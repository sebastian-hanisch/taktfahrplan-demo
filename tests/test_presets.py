"""Permalink, Presets und der Regler mit linienabhängigen Grenzen (Key trägt die Linienzahl)."""
import tkt_constants as C
import tkt_presets as PR


def test_budget_key_and_bounds_follow_the_line_count():
    assert PR.budget_key(6) == "budget_slider_6" and PR.budget_key(3) != PR.budget_key(10)
    assert PR.budget_bounds(8) == (0, 8) and PR.default_budget(3) == 3 and PR.default_budget(10) == C.DEFAULT_BUDGET


def test_snap_rounds_to_the_nearest_step_and_prefers_the_smaller_on_ties():
    assert PR.snap(C.LINES_OPTIONS, 7) == 6 and PR.snap(C.LINES_OPTIONS, 4) == 3 and PR.snap(C.LINES_OPTIONS, 99) == 10
    assert PR.snap(C.HEADWAY_OPTIONS, 1) == 0 and PR.snap(C.HEADWAY_OPTIONS, 2) == 3


def test_presets_cover_every_story_and_stay_outside_the_sweep_seeds():
    assert set(C.PRESET_ORDER) == set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESET_ORDER) == 5
    for name, p in C.PRESETS.items():
        assert p["lines"] in C.LINES_OPTIONS and p["headway"] in C.HEADWAY_OPTIONS and 0 <= p["budget"] <= p["lines"]
        assert C.SEED_MIN <= p["seed"] <= C.SEED_MAX
        for n in C.SWEEP_LINES:
            s0 = C.SWEEP_SEED_START[n]
            assert not (s0 <= p["seed"] < s0 + C.SWEEP_NETS), (name, n)       # Presets liegen nicht in der Stichprobe der Messreihe


def test_constants_agree_with_the_model_defaults():
    from tkt_timetable import Instance
    inst = Instance(C.T, [], {})
    assert (inst.dwell, inst.tmin, inst.transfer_min) == (C.DWELL, C.TURN_MIN, C.TRANSFER_MIN)
