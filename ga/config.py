"""Configuration and presets of the genetic algorithm."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace

SELECTION_METHODS = ("tournament", "roulette_rank")
FILL_STRATEGIES = ("random", "offspring")


@dataclass
class GAConfig:
    population: int = 500  # POB
    generations: int = 300
    selected_pct: float = 0.7  # SEL: share of the population selected as parents
    cross_pct: float = 0.7  # CRUCE: share of the parents used for crossover (the rest mutate)
    mut_catches: float = 0.15  # MUT_CAP: per-route probability of changing a capture
    mut_team: float = 0.15  # MUT_EQU: per-slot probability of changing a team member
    mut_order: float = 0.15  # per-team probability of swapping two positions (send-out order)
    survivors: int = 75  # SUP: unique elite individuals that survive
    immigrants_pct: float = 0.0  # share of new random individuals each generation (offspring strategy)
    selection: str = "roulette_rank"  # tournament | roulette_rank
    tournament_k: int = 3
    fill_strategy: str = "random"  # random: complete with random individuals (report); offspring: with children
    children_per_pair: int = 1
    smart_init: bool = True  # type-weighted gym teams for the random individuals
    switching: bool = False  # individuals carry a switch plan per battle (the report did not allow switching)
    mut_plan: float = 0.03  # per-entry probability of changing the switch plan
    plan_init: float = 0.1  # probability that a random individual plans an opponent for each rival Pokémon
    adaptive_modes: bool = False  # exploration / exploitation switching (as in the report experiments)
    exploration: dict = field(default_factory=lambda: {
        "selected_pct": 0.9, "cross_pct": 0.95, "mut_catches": 0.4, "mut_team": 0.4, "survivors_pct": 0.02})
    exploitation: dict = field(default_factory=lambda: {
        "selected_pct": 0.3, "cross_pct": 0.3, "mut_catches": 0.05, "mut_team": 0.05, "survivors_pct": 0.5})
    graded_loss_penalty: bool = False  # lost battle: +10000 + remaining rival HP (instead of +10000)
    # Random battles (accuracy, critical hits, damage rolls): every battle is played once per fixed
    # random scenario and its cost is the mean. 0 = deterministic simulator (as in the report).
    scenarios: int = 0
    scenario_seed: int = 0
    seed: int | None = None
    workers: int = 1
    checkpoint_every: int = 10
    preset: str = "personalizado"

    def validate(self) -> "GAConfig":
        if self.population < 4:
            raise ValueError("La población debe tener al menos 4 individuos")
        if self.generations < 1:
            raise ValueError("Debe haber al menos 1 generación")
        for name in ("selected_pct", "cross_pct", "mut_catches", "mut_team", "mut_order", "immigrants_pct",
                     "mut_plan", "plan_init"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1")
        if not 1 <= self.survivors <= self.population:
            raise ValueError("SUP debe estar entre 1 y el tamaño de la población")
        if self.selection not in SELECTION_METHODS:
            raise ValueError(f"Selección desconocida: {self.selection}")
        if self.fill_strategy not in FILL_STRATEGIES:
            raise ValueError(f"Estrategia de relleno desconocida: {self.fill_strategy}")
        if self.tournament_k < 2:
            raise ValueError("El torneo necesita k >= 2")
        if not 0 <= self.scenarios <= 100_000:
            raise ValueError("scenarios debe estar entre 0 (determinista) y 100000")
        if self.children_per_pair not in (1, 2):
            raise ValueError("children_per_pair debe ser 1 o 2")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GAConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_overrides(self, **kwargs) -> "GAConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


PRESETS = {
    # Best configuration of the report ("sin distinción" de modos): SEL 70 %, CRUCE 70 %,
    # MUT 15/15 %, SUP 75 and the population completed with random individuals.
    # The operators are the corrected ones and the roulette is rank based.
    "memoria": GAConfig(
        population=500, generations=300, selected_pct=0.7, cross_pct=0.7, mut_catches=0.15, mut_team=0.15,
        mut_order=0.15, survivors=75, immigrants_pct=0.0, selection="roulette_rank", fill_strategy="random",
        children_per_pair=1, smart_init=True, switching=False, preset="memoria"),
    # Classic generational scheme: tournament selection, the next population is made of
    # the elite plus the children, with a small share of random immigrants for diversity.
    "mejorado": GAConfig(
        population=500, generations=300, selected_pct=1.0, cross_pct=0.7, mut_catches=0.08, mut_team=0.08,
        mut_order=0.15, survivors=25, immigrants_pct=0.05, selection="tournament", tournament_k=3,
        fill_strategy="offspring", children_per_pair=2, smart_init=True, switching=True, mut_plan=0.03,
        plan_init=0.1, preset="mejorado"),
    # "mejorado" with random battles (accuracy, critical hits, damage rolls): 100 fixed scenarios per
    # battle, evaluated in parallel (the number of processes does not change the result).
    "aleatorio": GAConfig(
        population=500, generations=300, selected_pct=1.0, cross_pct=0.7, mut_catches=0.08, mut_team=0.08,
        mut_order=0.15, survivors=25, immigrants_pct=0.05, selection="tournament", tournament_k=3,
        fill_strategy="offspring", children_per_pair=2, smart_init=True, switching=True, mut_plan=0.03,
        plan_init=0.1, scenarios=100, workers=max(1, min(8, (os.cpu_count() or 2) - 1)), preset="aleatorio"),
    # Small and quick, for demos.
    "rapido": GAConfig(
        population=100, generations=50, selected_pct=1.0, cross_pct=0.7, mut_catches=0.08, mut_team=0.08,
        mut_order=0.15, survivors=6, immigrants_pct=0.05, selection="tournament", tournament_k=3,
        fill_strategy="offspring", children_per_pair=2, smart_init=True, switching=True, mut_plan=0.03,
        plan_init=0.1, preset="rapido"),
}

PRESET_LABELS = {
    "memoria": "Memoria (configuración del trabajo, sin cambios)",
    "mejorado": "Mejorado (torneo + élite + cambios)",
    "aleatorio": "Aleatorio (mejorado + combates con precisión, críticos y tiradas)",
    "rapido": "Rápido (demo)",
}


def preset(name: str) -> GAConfig:
    cfg = replace(PRESETS[name])
    cfg.exploration = dict(cfg.exploration)
    cfg.exploitation = dict(cfg.exploitation)
    return cfg
