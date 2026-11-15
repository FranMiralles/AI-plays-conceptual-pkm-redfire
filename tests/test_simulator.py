import json
import random
from pathlib import Path

import battle_simulator as bs
from ga import operators as ops
from ga.fitness import Evaluator

DATA = Path(__file__).parent / "data"


class _ZeroDamage(dict):
    def get(self, key, default=None):
        return ["tackle", 0.0]


def test_battle_without_damage_terminates_as_a_loss():
    dataset = dict(bs.dataset, damage_dictionary=_ZeroDamage())
    result = bs.simulate_battle([1, 4], bs.TRAINERS["GYM_PLATEADA"], [], [], dataset)
    assert result.timed_out and not result.won
    assert result.turns == bs.MAX_TURNS


def test_regression_against_original_simulator(damage_table_complete):
    """Fitness of the individuals stored in the old code. ``fitness_original`` is the value
    of the original simulator; the expected value includes the two simulator fixes
    (reset of charging state on switch and separate rival damage keys)."""
    cases = json.loads((DATA / "regression_individuals.json").read_text())
    for case in cases:
        _, fitness, _ = bs.calculate_fitness(case["genome"], bs.dataset)
        assert fitness == case["fitness"], case["name"]


def test_memoised_evaluation_matches_direct_simulation(damage_table_complete):
    rng = random.Random(9)
    evaluator = Evaluator()
    for _ in range(30):
        genome = ops.generate_individual(rng)
        ev = evaluator.evaluate(genome)
        _, direct, _ = bs.calculate_fitness(genome, bs.dataset)
        assert ev.fitness == direct
        assert evaluator.evaluate(genome) is ev  # genome cache


def test_events_reproduce_the_battle_result(damage_table_complete):
    genome = json.loads((DATA / "regression_individuals.json").read_text())[0]["genome"]
    for t in range(ops.N_TRAINERS):
        plain = bs.battle_for_trainer(genome, t, bs.dataset)
        detailed = bs.battle_for_trainer(genome, t, bs.dataset, record_events=True)
        assert (plain.won, plain.hp_lost, plain.turns) == (detailed.won, detailed.hp_lost, detailed.turns)
        end = detailed.events[-1]
        assert end["type"] == "end" and end["hp_lost"] == plain.hp_lost


def test_invalid_individual_gets_infinite_fitness(damage_table_complete):
    genome = json.loads((DATA / "regression_individuals.json").read_text())[1]["genome"]  # has repeated captures
    ev = Evaluator().evaluate(genome)
    assert not ev.valid and ev.fitness == float("inf") and ev.raw_fitness < float("inf")


# --------------------------------------------------------------------------- switching


def _plateada(team, plan=None, events=True):
    return bs.simulate_battle(team, bs.TRAINERS["GYM_PLATEADA"], [], [], bs.dataset, record_events=events, plan=plan)


def test_planned_switch_spends_the_turn_and_the_rival_hits_the_incoming_pokemon(damage_table_complete):
    # Charmander leads against Geodude; the plan sends Squirtle to face it
    result = _plateada([4, 7], plan=[7, None])
    first_turn = [e for e in result.events if e.get("turn") == 1]
    assert first_turn[1]["type"] == "switch" and first_turn[1]["to_id"] == 7
    actions = [e for e in first_turn if e["type"] == "action"]
    assert len(actions) == 1 and actions[0]["side"] == "rival" and actions[0]["defender"] == "Squirtle"
    assert result.switches == 1


def test_plan_that_names_the_lead_does_not_switch(damage_table_complete):
    plain = _plateada([7, 4], events=False)
    planned = _plateada([7, 4], plan=[7, 7], events=False)
    assert (plain.hp_lost, plain.turns, planned.switches) == (planned.hp_lost, planned.turns, 0)


def test_switching_to_a_counter_saves_hp(damage_table_complete):
    # Charmander, Pidgey and Squirtle against Brock: without plan Charmander and Pidgey fall to Onix;
    # planning Squirtle for Onix switches it in as soon as Onix comes out.
    plain = _plateada([4, 16, 7], events=False)
    planned = _plateada([4, 16, 7], plan=[None, 7])
    assert planned.switches == 1 and planned.won
    assert planned.hp_lost < plain.hp_lost
    switch = next(e for e in planned.events if e["type"] == "switch")
    assert switch["r_species"] == "Onix" and switch["to_id"] == 7


def test_memo_distinguishes_plans(damage_table_complete):
    import random as _random
    rng = _random.Random(4)
    genome = ops.generate_individual(rng, switching=True)
    no_plan = [genome[0], genome[1], ops.empty_plans()]
    ev = Evaluator()
    assert ev.evaluate(genome).fitness == bs.calculate_fitness(genome, bs.dataset)[1]
    assert ev.evaluate(no_plan).fitness == bs.calculate_fitness(no_plan, bs.dataset)[1]
