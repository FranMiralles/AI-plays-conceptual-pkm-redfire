"""Fitness evaluation with memoisation.

The fitness is decomposable: each battle only depends on the trainer, the
player's team (and order) and, for the rival battles, the starter. Children
inherit whole teams from their parents, so most battles are cache hits.

``Evaluator`` uses the deterministic simulator. ``RandomEvaluator`` plays every battle once per
random scenario (stochastic_battle.py) and the cost of a battle is its mean over the scenarios;
the scenarios are fixed, so the fitness is still deterministic and cacheable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import battle_simulator as bs
from ga.operators import N_TRAINERS, genome_key, violations


@dataclass(frozen=True)
class BattleSummary:
    won: bool
    hp_lost: int
    rival_hp_left: int
    turns: int
    timed_out: bool
    switches: int = 0
    # random battles (mean over the scenarios); None with the deterministic simulator
    win_rate: float | None = None
    cost: float | None = None  # mean of: % HP lost if won, LOSS_PENALTY if lost
    cost_graded: float | None = None  # same with LOSS_PENALTY + rival HP left


@dataclass
class Evaluation:
    fitness: float  # value minimised by the GA (inf if the individual breaks a constraint)
    raw_fitness: float  # battles only (inf if a team uses Pokémon not caught yet)
    valid: bool
    violations: list = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    hp_lost: int = 0
    per_battle: list = field(default_factory=list)  # list[BattleSummary]
    p_complete: float | None = None  # random battles: probability of winning the 21 battles

    @property
    def completes_game(self) -> bool:
        return self.valid and self.losses == 0 and len(self.per_battle) == N_TRAINERS

    def per_battle_values(self) -> list:
        """HP lost per battle, ``-1`` for a lost battle (compact form for charts)."""
        return [b.hp_lost if b.won else -1 for b in self.per_battle]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fitness"] = None if self.fitness == float("inf") else self.fitness
        d["raw_fitness"] = None if self.raw_fitness == float("inf") else self.raw_fitness
        return d


class Evaluator:
    def __init__(self, dataset: dict | None = None, graded_loss_penalty: bool = False,
                 battle_cache_size: int = 400_000, genome_cache_size: int = 100_000):
        self.dataset = dataset if dataset is not None else bs.dataset
        self.graded_loss_penalty = graded_loss_penalty
        self.battle_cache_size = battle_cache_size
        self.genome_cache_size = genome_cache_size
        self._battles: dict = {}
        self._genomes: dict = {}
        self.evaluations = 0
        self.genome_hits = 0
        self.battle_hits = 0
        self.battle_misses = 0

    # ------------------------------------------------------------------ caches

    @staticmethod
    def _trim(cache: dict, limit: int) -> None:
        if len(cache) > limit:
            for _ in range(len(cache) - int(limit * 0.9)):
                del cache[next(iter(cache))]

    def battle(self, genome, trainer_index: int) -> BattleSummary:
        key = battle_key(genome, trainer_index)
        summary = self._battles.get(key)
        if summary is not None:
            self.battle_hits += 1
            return summary
        self.battle_misses += 1
        r = bs.battle_for_trainer(genome, trainer_index, self.dataset)
        summary = BattleSummary(r.won, r.hp_lost, r.rival_hp_left, r.turns, r.timed_out, r.switches)
        self._battles[key] = summary
        self._trim(self._battles, self.battle_cache_size)
        return summary

    # ------------------------------------------------------------------ fitness

    def battle_cost(self, b: BattleSummary) -> float:
        if b.cost is not None:
            return b.cost_graded if self.graded_loss_penalty else b.cost
        if b.won:
            return b.hp_lost
        return bs.LOSS_PENALTY + (b.rival_hp_left if self.graded_loss_penalty else 0)

    def evaluate(self, genome) -> Evaluation:
        self.evaluations += 1
        key = genome_key(genome)
        cached = self._genomes.get(key)
        if cached is not None:
            self.genome_hits += 1
            return cached
        problems = violations(genome)
        if not bs.teams_use_available_pokemon(genome):
            ev = Evaluation(float("inf"), float("inf"), False, problems)
        else:
            per_battle = [self.battle(genome, t) for t in range(N_TRAINERS)]
            raw = sum(self.battle_cost(b) for b in per_battle)
            wins = sum(b.won for b in per_battle)
            p_complete = None
            if per_battle and per_battle[0].win_rate is not None:
                raw = round(raw, 1)
                p_complete = 1.0
                for b in per_battle:
                    p_complete *= b.win_rate
                p_complete = round(p_complete, 4)
            ev = Evaluation(
                fitness=raw if not problems else float("inf"),
                raw_fitness=raw,
                valid=not problems,
                violations=problems,
                wins=wins,
                losses=len(per_battle) - wins,
                hp_lost=sum(b.hp_lost for b in per_battle),
                per_battle=per_battle,
                p_complete=p_complete,
            )
        self._genomes[key] = ev
        self._trim(self._genomes, self.genome_cache_size)
        return ev

    def battle_cache_hit_rate(self) -> float:
        total = self.battle_hits + self.battle_misses
        return self.battle_hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "genome_hits": self.genome_hits,
            "battles_simulated": self.battle_misses,
            "battle_hit_rate": round(self.battle_cache_hit_rate(), 4),
        }


def battle_key(genome, trainer_index: int) -> tuple:
    catches, teams = genome[0], genome[1]
    plan = genome[2][trainer_index] if len(genome) > 2 else None
    starter = catches[0] if trainer_index in bs.RIVAL_TRAINER_INDICES else None
    plan_key = tuple(plan) if plan and any(x is not None for x in plan) else ()
    return (trainer_index, starter, tuple(teams[trainer_index]), plan_key)


class RandomEvaluator(Evaluator):
    """Fitness = mean cost of every battle over ``scenarios`` fixed random scenarios.

    ``prefetch(genomes)`` simulates all the battles missing from the cache in one vectorised
    batch, so call it before evaluating a population; ``battle`` falls back to simulating
    the missing battles of one genome."""

    def __init__(self, scenarios: int = 100, scenario_seed: int = 0, dataset: dict | None = None, **kw):
        super().__init__(dataset=dataset, **kw)
        import stochastic_battle as sb

        self._sb = sb
        self.store = sb.default_store()
        self.scenarios = sb.Scenarios(scenarios, seed=scenario_seed)

    def simulate_items(self, items) -> list:
        """Summaries of the battles ``[(genome, trainer_index), ...]`` (no cache)."""
        sb = self._sb
        specs = [sb.compile_battle(g, t, self.store, self.dataset) for g, t in items]
        res = sb.simulate(specs, self.scenarios, self.store)
        return [self._summary(res, i) for i in range(len(items))]

    def prefetch(self, genomes, map_chunks=None, n_chunks: int = 1) -> int:
        """Simulates the battles of ``genomes`` that are not cached. Returns how many.

        ``map_chunks(list_of_item_chunks) -> iterable of summary lists`` spreads the work over
        processes (the cache stays in this process, so hits are not lost between generations)."""
        todo = {}
        for g in genomes:
            if genome_key(g) in self._genomes or not bs.teams_use_available_pokemon(g):
                continue
            for t in range(N_TRAINERS):
                key = battle_key(g, t)
                if key not in self._battles and key not in todo:
                    todo[key] = (g, t)
        if not todo:
            return 0
        items = list(todo.values())
        if map_chunks is not None and n_chunks > 1 and len(items) > 50:
            size = -(-len(items) // n_chunks)
            summaries = [sm for part in map_chunks([items[i:i + size] for i in range(0, len(items), size)])
                         for sm in part]
        else:
            summaries = self.simulate_items(items)
        for key, summary in zip(todo, summaries):
            self._battles[key] = summary
        self.battle_misses += len(todo)
        self.battle_hits -= len(todo)  # battle() counts them again as hits
        self._trim(self._battles, self.battle_cache_size)
        return len(todo)

    @staticmethod
    def _summary(res, i: int) -> BattleSummary:
        import numpy as np

        won = res.won[i]
        lost_hp = res.hp_lost[i].astype(np.float64)
        cost = np.where(won, lost_hp, bs.LOSS_PENALTY)
        graded = np.where(won, lost_hp, bs.LOSS_PENALTY + res.rival_hp_left[i])
        win_rate = float(won.mean())
        return BattleSummary(won=win_rate >= 0.5, hp_lost=int(round(float(lost_hp.mean()))),
                             rival_hp_left=int(round(float(res.rival_hp_left[i].mean()))),
                             turns=int(round(float(res.turns[i].mean()))), timed_out=bool(res.timed_out[i].all()),
                             switches=int(round(float(res.switches[i].mean()))), win_rate=round(win_rate, 4),
                             cost=round(float(cost.mean()), 2), cost_graded=round(float(graded.mean()), 2))

    def battle(self, genome, trainer_index: int) -> BattleSummary:
        key = battle_key(genome, trainer_index)
        if key not in self._battles:
            sb = self._sb
            res = sb.simulate([sb.compile_battle(genome, trainer_index, self.store, self.dataset)],
                              self.scenarios, self.store)
            self._battles[key] = self._summary(res, 0)
            self.battle_misses += 1
            self._trim(self._battles, self.battle_cache_size)
            return self._battles[key]
        self.battle_hits += 1
        return self._battles[key]


def make_evaluator(cfg) -> Evaluator:
    """Evaluator for a GAConfig: deterministic, or random battles if ``cfg.scenarios > 0``."""
    scenarios = getattr(cfg, "scenarios", 0) or 0
    if scenarios > 0:
        return RandomEvaluator(scenarios=scenarios, scenario_seed=getattr(cfg, "scenario_seed", 0),
                               graded_loss_penalty=cfg.graded_loss_penalty)
    return Evaluator(graded_loss_penalty=cfg.graded_loss_penalty)


def simulate_detailed(genome, dataset: dict | None = None) -> list:
    """Full simulation with turn-by-turn events for the 21 battles (for the viewer)."""
    dataset = dataset if dataset is not None else bs.dataset
    return [bs.battle_for_trainer(genome, t, dataset, record_events=True) for t in range(N_TRAINERS)]
