import random

from ga.config import GAConfig, preset
from ga.engine import GeneticAlgorithm, Individual, read_history


class _Fake:
    def __init__(self, fitness):
        self.fitness = fitness


def _old_roulette_probabilities(fitness):
    """Selection probabilities of the original implementation (1 - f/sum, normalised)."""
    total = sum(fitness)
    probs = [1 - f / total for f in fitness]
    s = sum(probs)
    return [p / s for p in probs]


def test_original_roulette_was_almost_uniform():
    fitness = list(range(2000, 2000 + 500 * 60, 60))  # 500 individuals from 2000 to ~32000
    probs = _old_roulette_probabilities(fitness)
    assert max(probs) / min(probs) < 1.03  # the best is only ~2 % more likely than the worst


def _selection_pressure(selection: str) -> float:
    cfg = GAConfig(population=500, selection=selection, seed=1)
    ga = GeneticAlgorithm(cfg)
    pop = [_Fake(f) for f in random.Random(0).sample(range(2000, 40000), 500)]
    chosen = ga._select_parents(pop, 350)
    best_decile = sorted(p.fitness for p in pop)[50]
    return sum(p.fitness <= best_decile for p in chosen) / len(chosen)


def test_new_selection_methods_prefer_good_individuals():
    assert _selection_pressure("tournament") > 0.2  # ~0.27 expected vs 0.10 at random
    assert _selection_pressure("roulette_rank") > 0.12


def _tiny_config(**kw):
    cfg = preset("rapido")
    cfg.population, cfg.generations, cfg.survivors, cfg.seed, cfg.checkpoint_every = 20, 6, 3, 42, 3
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_same_seed_same_result(damage_table_complete):
    a = GeneticAlgorithm(_tiny_config()).run()
    b = GeneticAlgorithm(_tiny_config()).run()
    assert a["best"]["genome"] == b["best"]["genome"]


def test_elitism_never_loses_the_best(damage_table_complete):
    bests = []
    GeneticAlgorithm(_tiny_config(generations=10), on_generation=lambda s, _: bests.append(s["best"])).run()
    assert all(b2 <= b1 for b1, b2 in zip(bests, bests[1:]))


def test_population_size_is_constant(damage_table_complete):
    for fill in ("random", "offspring"):
        sizes = []
        GeneticAlgorithm(_tiny_config(fill_strategy=fill),
                         on_generation=lambda s, ga: sizes.append(len(ga.population))).run()
        assert set(sizes) == {20}


def test_resume_continues_the_same_trajectory(tmp_path, damage_table_complete):
    full = GeneticAlgorithm(_tiny_config(), run_dir=tmp_path / "full").run()

    stop_at = {"gen": None}

    def stop_after_checkpoint(stats, _ga):
        if stats["gen"] == 3:
            stop_at["gen"] = 3

    partial = GeneticAlgorithm(_tiny_config(), run_dir=tmp_path / "part", on_generation=stop_after_checkpoint,
                               should_stop=lambda: stop_at["gen"] is not None)
    partial.run()
    assert partial.generation == 3
    resumed = GeneticAlgorithm.resume(tmp_path / "part").run()
    assert resumed["best"]["genome"] == full["best"]["genome"]
    assert [h["gen"] for h in read_history(tmp_path / "part")] == list(range(7))


def test_individual_display_origin():
    ind = Individual(genome=[[], []], origin="crossover", born=3)
    assert ind.display_origin(3) == "crossover" and ind.display_origin(4) == "elite"


def test_switching_run_is_reproducible_and_valid(damage_table_complete):
    from ga import operators as ops

    cfg = _tiny_config()
    cfg.switching, cfg.plan_init, cfg.mut_plan = True, 0.4, 0.2
    a = GeneticAlgorithm(cfg).run()
    cfg2 = _tiny_config()
    cfg2.switching, cfg2.plan_init, cfg2.mut_plan = True, 0.4, 0.2
    b = GeneticAlgorithm(cfg2).run()
    assert a["best"]["genome"] == b["best"]["genome"]
    assert ops.violations(a["best"]["genome"]) == []
    assert len(a["best"]["genome"]) == 3
