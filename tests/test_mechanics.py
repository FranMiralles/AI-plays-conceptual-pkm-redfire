"""Generation 3 mechanics of the random battle simulator, checked on recorded battles."""
import random

import numpy as np
import pytest

import battle_simulator as bs
import stochastic_battle as sb
from ga import operators as ops

SCENARIOS = 6
GENOMES = 14


@pytest.fixture(scope="module")
def store():
    try:
        return sb.default_store()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def recorded(store):
    """Events of GENOMES random individuals x 21 battles x SCENARIOS scenarios."""
    rng = random.Random(7)
    genomes = [ops.generate_individual(rng, smart=True, pkdex=bs.dataset["pkdex"], switching=True, plan_prob=0.2)
               for _ in range(GENOMES)]
    specs = [sb.compile_battle(g, t, store) for g in genomes for t in range(ops.N_TRAINERS)]
    res = sb.simulate(specs, sb.Scenarios(SCENARIOS, seed=11), store, record=True)
    plain = sb.simulate(specs, sb.Scenarios(SCENARIOS, seed=11), store)
    battles = [(specs[b], res.events[b][s]) for b in range(len(specs)) for s in range(SCENARIOS)]
    return {"specs": specs, "res": res, "plain": plain, "battles": battles}


def _events(recorded, etype=None):
    for spec, evs in recorded["battles"]:
        for n, ev in enumerate(evs):
            if etype is None or ev["type"] == etype:
                yield spec, evs, n, ev


def test_recording_does_not_change_the_battles(recorded):
    res, plain = recorded["res"], recorded["plain"]
    for field in ("won", "hp_lost", "rival_hp_left", "turns", "switches", "timed_out"):
        assert np.array_equal(getattr(res, field), getattr(plain, field)), field


def test_replay_matches_the_evaluation(store):
    rng = random.Random(3)
    genome = ops.generate_individual(rng, smart=True, pkdex=bs.dataset["pkdex"], switching=True, plan_prob=0.2)
    scores = sb.score_genomes([genome], 20, store=store)
    for s in (0, 7, 19):
        battles = sb.replay(genome, s, store=store)
        fit = sum(b.hp_lost if b.won else bs.LOSS_PENALTY for b in battles)
        assert abs(fit - scores.fitness[0, s]) <= 21  # replay rounds the HP lost of each battle


def test_mechanics_happen(recorded):
    kinds = {ev["type"] for _, _, _, ev in _events(recorded)}
    statuses = {ev["status"] for _, _, _, ev in _events(recorded, "status")}
    reasons = {ev["reason"] for _, _, _, ev in _events(recorded, "cant_move")}
    assert {"status", "stat", "confuse", "residual", "ability", "cant_move"} <= kinds
    assert {"par", "brn", "psn"} <= statuses
    assert {"paralysis", "flinch"} <= reasons


def test_full_paralysis_is_one_in_four(recorded):
    tried = stopped = 0
    for _, evs, n, ev in _events(recorded):
        if ev["type"] == "action" and ev.get("kind") == "attack":
            side = ev["side"]
            if (ev["p_status"] if side == "player" else ev["r_status"]) == "par":
                tried += 1
        if ev["type"] == "cant_move" and ev["reason"] == "paralysis":
            stopped += 1
            tried += 1
    assert tried > 150
    assert 0.17 < stopped / tried < 0.33


def test_status_immunities(recorded, store):
    from ga.damage_table import load_rolls

    species = load_rolls()["__species__"]
    for spec, evs, n, ev in _events(recorded, "status"):
        name = ev["pokemon"]
        types = species[name]["types"]
        if ev["status"] == "brn":
            assert "fire" not in types
        if ev["status"] == "frz":
            assert "ice" not in types
        if ev["status"] in ("psn", "tox"):
            assert "poison" not in types and "steel" not in types


def test_ability_immunities(recorded):
    blocked = {"par": "limber", "psn": "immunity", "tox": "immunity", "brn": "water-veil", "frz": "magma-armor",
               "slp": "insomnia"}
    for spec, evs, n, ev in _events(recorded, "status"):
        side = 0 if ev["side"] == "player" else 1
        idx = ev["p_idx"] if side == 0 else ev["r_idx"]
        mask = (spec.p_ability if side == 0 else spec.r_ability)[idx]
        assert not mask & sb.ABILITY_BITS[blocked[ev["status"]]]
    for spec, evs, n, ev in _events(recorded, "confuse"):
        side = 0 if ev["side"] == "player" else 1
        idx = ev["p_idx"] if side == 0 else ev["r_idx"]
        assert not (spec.p_ability if side == 0 else spec.r_ability)[idx] & sb.ABILITY_BITS["own-tempo"]


def test_residual_damage_amounts(recorded):
    """Burn and poison: 1/8 of the max HP; bad poison: n/16 growing every turn; trapping: 1/16."""
    seen = set()
    for spec, evs, n, ev in _events(recorded, "residual"):
        seen.add(ev["cause"])
        expected = {"brn": 12.5, "psn": 12.5, "trap": 6.25}.get(ev["cause"])
        if expected is not None and not ev["fainted"]:
            assert abs(ev["amount"] - expected) <= 1.5, ev
    assert {"brn", "psn"} & seen


def test_stat_stages_stay_in_range_and_rock_tomb_always_lowers_speed(recorded):
    hits = lowered = 0
    for spec, evs, n, ev in _events(recorded, "stat"):
        assert -6 <= ev["delta"] <= 6
    for spec, evs, n, ev in _events(recorded, "action"):
        if ev["move"] == "rock-tomb" and not ev["missed"] and ev["damage"] > 0 and not ev["fainted"]:
            target_side = 1 if ev["side"] == "player" else 0
            idx = ev["r_idx"] if target_side else ev["p_idx"]
            if (spec.r_ability if target_side else spec.p_ability)[idx] & sb.ABILITY_BITS["shield-dust"]:
                continue  # Shield Dust blocks secondary effects
            hits += 1
            nxt = evs[n + 1] if n + 1 < len(evs) else {}
            lowered += nxt.get("type") == "stat" and nxt.get("stat") == "spe"
    if hits:
        assert lowered == hits


def test_flinch_only_when_moving_second(recorded):
    for spec, evs, n, ev in _events(recorded, "cant_move"):
        if ev["reason"] != "flinch":
            continue
        # the opponent acted earlier in the same turn
        k = n - 1
        while k >= 0 and evs[k]["type"] != "turn":
            k -= 1
        acted = [e for e in evs[k + 1:n] if e["type"] == "action" and e["side"] != ev["side"]]
        assert acted


def test_confusion_self_hit_is_half(recorded):
    hit = calm = 0
    for spec, evs, n, ev in _events(recorded):
        hit += ev["type"] == "confusion_hit"
        calm += ev["type"] == "confused"
    assert hit + calm > 60
    assert 0.35 < hit / (hit + calm) < 0.65


def test_damage_formula_matches_the_calculator(store):
    """Every move whose damage is recomputed reproduces @smogon/calc at stage 0 (checked when the
    store is built); tools/validate_damage.py checks stages, burns and critical hits."""
    assert store.validation["formula"] > 60000
    assert store.validation["table"] <= 5


def test_stage_and_damage_helpers():
    assert sb._stage(100, 1) == 150 and sb._stage(100, -1) == 66 and sb._stage(100, 6) == 400
    assert sb._stage(100, -6) == 25
    # level 50, 100 attack, 80 power, 100 defence, physical, STAB, neutral
    plain = sb.gen3_damage(50, 100, 80, 100, True, True, 4)
    assert plain == ((22 * 100 * 80 // 100) // 50 + 2) * 3 // 2
    assert sb.gen3_damage(50, 100, 80, 100, True, True, 4, burned=True) < plain
    assert sb.gen3_damage(5, 5, 10, 300, True, False, 1) == 1  # 3 x 1/4 rounds to 0: at least 1 unless immune
    assert sb.gen3_damage(50, 100, 80, 100, True, False, 0) == 0
