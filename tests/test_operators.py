import random

import pytest

import battle_simulator as bs
from ga import operators as ops


@pytest.fixture(scope="module")
def population():
    rng = random.Random(11)
    return [ops.generate_individual(rng, smart=i % 2 == 0, pkdex=bs.dataset["pkdex"]) for i in range(40)]


def test_generated_individuals_are_feasible(population):
    for genome in population:
        assert ops.violations(genome) == []
        assert ops.teams_complete(genome)


def test_operators_keep_feasibility(population):
    rng = random.Random(3)
    pop = [list(g) for g in population]
    for it in range(5000):
        a, b = rng.sample(pop, 2)
        child = ops.crossover(a, b, rng) if it % 2 else ops.mutate(a, rng, 0.3, 0.3, 0.3)
        assert ops.violations(child) == [], ops.violations(child)
        assert ops.teams_complete(child)
        pop[rng.randrange(len(pop))] = child


def test_crossover_never_repeats_captures(population):
    rng = random.Random(5)
    for _ in range(2000):
        a, b = rng.sample(population, 2)
        catches = [p for p in ops.crossover(a, b, rng)[0] if p is not None]
        assert len(catches) == len(set(catches))


def test_mutation_without_probability_keeps_genome_and_order(population):
    rng = random.Random(1)
    for genome in population:
        assert ops.mutate(genome, rng, 0.0, 0.0, 0.0) == genome


def test_mutation_does_not_reorder_untouched_teams(population):
    """The original mutation reordered every team with list(set(...))."""
    rng = random.Random(2)
    genome = population[0]
    mutated = ops.mutate(genome, rng, 0.0, 0.0, 0.0)
    for t in range(ops.N_TRAINERS):
        assert mutated[1][t] == genome[1][t]


def test_crossover_keeps_league_permutations(population):
    rng = random.Random(4)
    for _ in range(500):
        a, b = rng.sample(population, 2)
        child = ops.crossover(a, b, rng)
        league = child[1][ops.LEAGUE_FIRST:]
        assert len({tuple(sorted(t)) for t in league}) == 1
        # if the league team needed no repair, its orders are inherited from one parent
        for parent in (a, b):
            if child[1][ops.LEAGUE_FIRST] == parent[1][ops.LEAGUE_FIRST]:
                assert child[1][ops.LEAGUE_FIRST:] == parent[1][ops.LEAGUE_FIRST:]


def test_crossover_of_identical_parents_is_identity(population):
    rng = random.Random(6)
    for genome in population[:10]:
        assert ops.crossover(genome, genome, rng) == genome


def test_violations_detect_repeated_captures():
    genome = ops.generate_individual(random.Random(0))
    catches = genome[0]
    # put the RUTA 1 capture again in RUTA 2 (both routes contain Pidgey and Rattata)
    i1, i2 = ops.ROUTES_ORDER.index("RUTA 1"), ops.ROUTES_ORDER.index("RUTA 2")
    catches[i2] = catches[i1]
    assert any("repetida" in v for v in ops.violations(genome))


def test_deadline_matching_places_conflicting_pokemon():
    """Pidgey (16), Rattata (19), Spearow (21) and Caterpie (10) are all needed before the
    first gym. Before it, Spearow only appears in RUTA 22. The original greedy assignment
    ("earliest free route", iterating a set: 16, 10, 19, 21) put Rattata in RUTA 22 and
    lost Spearow; the matching places the four of them."""
    rng = random.Random(0)
    deadline = ops.CUT[1]  # GYM PLATEADA
    route_of = ops._assign_required({16: deadline, 19: deadline, 21: deadline, 10: deadline}, {}, rng)
    assert sorted(route_of.values()) == [10, 16, 19, 21]
    assert route_of[ops.ROUTES_ORDER.index("RUTA 22")] == 21


# --------------------------------------------------------------------------- switch plans


@pytest.fixture(scope="module")
def population_with_plans():
    rng = random.Random(21)
    return [ops.generate_individual(rng, smart=i % 2 == 0, pkdex=bs.dataset["pkdex"], switching=True)
            for i in range(40)]


def test_plans_have_one_entry_per_rival_pokemon(population_with_plans):
    for genome in population_with_plans:
        assert [len(p) for p in genome[2]] == list(ops.RIVAL_SIZES)
        assert ops.violations(genome) == []
    assert any(ops.has_switching(g) for g in population_with_plans)


def test_operators_keep_plans_valid(population_with_plans):
    rng = random.Random(8)
    pop = list(population_with_plans)
    for it in range(4000):
        a, b = rng.sample(pop, 2)
        child = ops.crossover(a, b, rng) if it % 2 else ops.mutate(a, rng, 0.3, 0.3, 0.3, 0.3)
        assert ops.violations(child) == [], ops.violations(child)
        pop[rng.randrange(len(pop))] = child


def test_plan_follows_team_member_replacement():
    old_teams = [[1, 2, 3]] + [[]] * (ops.N_TRAINERS - 1)
    new_teams = [[1, 9, 3]] + [[]] * (ops.N_TRAINERS - 1)
    plans = [[2, None, 3]] + [[] for _ in range(ops.N_TRAINERS - 1)]
    ops.remap_plans(plans, old_teams, new_teams)
    assert plans[0] == [9, None, 3]  # the newcomer takes the role of the Pokémon it replaced


def test_old_two_part_genomes_are_normalized():
    genome = ops.generate_individual(random.Random(0))[:2]
    normalized = ops.normalize(genome)
    assert normalized[:2] == genome and not ops.has_switching(normalized)
    assert ops.genome_key(genome) == ops.genome_key(normalized)
