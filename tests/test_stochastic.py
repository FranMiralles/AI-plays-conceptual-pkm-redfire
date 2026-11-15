import json
import random
from pathlib import Path

import numpy as np
import pytest

import battle_simulator as bs
import stochastic_battle as sb
from ga import operators as ops

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rolls_store():
    try:
        return sb.default_store()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _random_genomes(n, seed=5, plan_prob=0.5):
    rng = random.Random(seed)
    return [ops.generate_individual(rng, smart=True, pkdex=bs.dataset["pkdex"], switching=True, plan_prob=plan_prob)
            for _ in range(n)]


def test_compat_mode_replays_the_deterministic_simulator(damage_table_complete):
    """With the deterministic damage table, no randomness and the deterministic quirks, the
    vectorised engine gives exactly the same battles (switches, two-turn moves, recharge...)."""
    store = sb.MoveStore.from_deterministic(bs.dataset["damage_dictionary"], bs.dataset["moves"])
    genomes = _random_genomes(40)
    specs, expected = [], []
    for g in genomes:
        for t in range(ops.N_TRAINERS):
            specs.append(sb.compile_battle(g, t, store))
            r = bs.battle_for_trainer(g, t, bs.dataset)
            expected.append((r.won, r.hp_lost, r.rival_hp_left, r.turns, r.switches, r.timed_out))
    res = sb.simulate(specs, sb.Scenarios(1), store, sb.COMPAT_RULES)
    got = [(bool(res.won[i, 0]), int(res.hp_lost[i, 0]), int(res.rival_hp_left[i, 0]), int(res.turns[i, 0]),
            int(res.switches[i, 0]), bool(res.timed_out[i, 0])) for i in range(len(specs))]
    assert got == expected
    assert sum(e[4] for e in expected) > 100  # the sample does exercise switching


def test_scenarios_are_reproducible_and_nested():
    a, b = sb.Scenarios(50, seed=3), sb.Scenarios(50, seed=3)
    assert np.array_equal(a.block, b.block)
    big = sb.Scenarios(200, seed=3)
    assert np.array_equal(big.block[:, :50], a.block)  # scenario s is the same in any set size
    assert np.array_equal(sb.Scenarios(150, seed=3, start=50).block, big.block[:, 50:])
    assert not np.array_equal(sb.Scenarios(50, seed=4).block, a.block)


def test_random_numbers_follow_the_game_probabilities():
    block = sb.Scenarios(400, seed=1).block.astype(np.int64)
    acc = (block[..., sb.S_ACC] * 100) >> 16
    assert abs((acc < 70).mean() - 0.70) < 0.01  # Thunder hits 70 % of the time
    assert abs((block[..., sb.S_CRIT] < sb.CRIT_THRESHOLD[0]).mean() - 1 / 16) < 0.005
    hits = sb.HITS_2_5[block[..., sb.S_HITS] >> 13]
    assert abs(hits.mean() - 3.0) < 0.02 and abs((hits == 2).mean() - 0.375) < 0.01
    rolls = np.bincount((block[..., sb.S_ROLL] >> 12).ravel(), minlength=16) / block[..., 0].size
    assert np.allclose(rolls, 1 / 16, atol=0.003)


def test_policy_prefers_a_sure_ko_with_the_most_accurate_move():
    rolls = {"Pikachu|20|Pidgey|20": [60, [["thunder", 60, 120, False], ["thunderbolt", 50, 100, False],
                                           ["quick-attack", 20, 40, False]]]}
    store = sb.MoveStore.from_rolls(rolls, bs.dataset["moves"])
    names = {v: k for k, v in store.move_ids.items()}
    choice = lambda hp: names[int(store["move_id"][0, store["policy"][0, hp]])]  # noqa: E731
    assert choice(10) == "quick-attack"  # every move KOs: 100 % accuracy and priority
    assert choice(40) == "thunderbolt"  # lowest roll of Thunderbolt is 42: sure KO with 100 % accuracy
    assert choice(51) == "thunder"  # only Thunder (lowest roll 51) KOs for sure
    assert choice(60) == "thunderbolt"  # no sure KO: best expected damage (~49 vs ~59 x 0.7 = ~41)


def test_random_fitness_is_deterministic_and_matches_single_battles(rolls_store):
    from ga.fitness import RandomEvaluator

    genomes = _random_genomes(6, seed=11, plan_prob=0.2)
    a = RandomEvaluator(scenarios=64, scenario_seed=2)
    a.prefetch(genomes)
    fa = [a.evaluate(g).fitness for g in genomes]
    b = RandomEvaluator(scenarios=64, scenario_seed=2)
    fb = [b.evaluate(g).fitness for g in genomes]  # no prefetch: battle by battle
    assert fa == fb
    c = RandomEvaluator(scenarios=64, scenario_seed=3)
    assert [c.evaluate(g).fitness for g in genomes] != fa


def test_example_champion_is_robust_but_not_perfect(rolls_store):
    from ga.engine import load_individual
    from ga.fitness import RandomEvaluator

    examples = json.loads((ROOT / "examples" / "individuals.json").read_text(encoding="utf-8"))
    genome = load_individual(examples[0])
    ev = RandomEvaluator(scenarios=100).evaluate(genome)
    assert ev.valid and 0.9 < ev.p_complete <= 1.0
    assert ev.fitness > examples[0]["fitness"]  # misses and rolls cost HP compared with the median roll
