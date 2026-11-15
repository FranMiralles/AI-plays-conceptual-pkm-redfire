"""Genetic algorithm engine.

Generation loop (see the report, chapter 3, and ``GAConfig``):
  1. select parents (rank roulette or tournament),
  2. crossover / mutation of the parents (each new individual remembers its origin),
  3. evaluate the new individuals,
  4. survivors: the ``survivors`` best *unique* individuals of parents + children,
  5. complete the population with random individuals (``fill_strategy=random``, as in
     the report) or with the best children plus a few immigrants (``offspring``).

Every generation produces a statistics dict (``on_generation``); with a ``run_dir``
the run is saved (config, history, hall of fame and checkpoints to resume it).
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import battle_simulator as bs
from ga import operators as ops
from ga.config import GAConfig
from ga.fitness import Evaluation, Evaluator, make_evaluator

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
ORIGINS = ("initial", "elite", "crossover", "mutation", "immigrant")
HALL_OF_FAME_SIZE = 50
POPULATION_SAMPLE_UI = 300
POPULATION_SAMPLE_HISTORY = 150
INF = float("inf")


class StopRequested(Exception):
    pass


@dataclass
class Individual:
    genome: list
    origin: str
    born: int
    ev: Evaluation | None = None
    _key: tuple | None = None

    @property
    def key(self) -> tuple:
        if self._key is None:
            self._key = ops.genome_key(self.genome)
        return self._key

    @property
    def fitness(self) -> float:
        return INF if self.ev is None else self.ev.fitness

    def display_origin(self, generation: int) -> str:
        """Origin as seen in a generation: individuals from previous generations are survivors."""
        return "elite" if self.born < generation else self.origin


def _finite(x: float):
    return None if x is None or math.isinf(x) else x


def _percentile(sorted_values: list, q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


# --------------------------------------------------------------------------- workers

_worker_evaluator: Evaluator | None = None


def _init_worker(config: dict):
    global _worker_evaluator
    _worker_evaluator = make_evaluator(GAConfig.from_dict(config))


def _simulate_items(items: list) -> list:
    return _worker_evaluator.simulate_items(items)


def _evaluate_chunk(genomes: list) -> list:
    if hasattr(_worker_evaluator, "prefetch"):
        _worker_evaluator.prefetch(genomes)
    return [_worker_evaluator.evaluate(g) for g in genomes]


# --------------------------------------------------------------------------- engine


class GeneticAlgorithm:
    def __init__(self, config: GAConfig, run_dir: Path | str | None = None, on_generation=None,
                 should_stop=None, wait_if_paused=None, log=print):
        self.cfg = config.validate()
        self.run_dir = Path(run_dir) if run_dir else None
        self.on_generation = on_generation
        self.should_stop = should_stop or (lambda: False)
        self.wait_if_paused = wait_if_paused or (lambda: None)
        self.log = log
        seed = self.cfg.seed if self.cfg.seed is not None else random.SystemRandom().randrange(2 ** 31)
        self.cfg.seed = seed
        self.rng = random.Random(seed)
        self.stats_rng = random.Random(seed + 1)  # sampling for statistics, independent of the search
        self.evaluator = make_evaluator(self.cfg)
        self.pool = None
        self.population: list[Individual] = []
        self.generation = 0
        self.best: Individual | None = None
        self.best_found_gen = 0
        self.best_history: list[float] = []
        self.mode_history: list[str] = []
        self.mode: str | None = None
        self.improvements: list[dict] = []
        self.hall_of_fame: dict = {}
        self.elapsed_before = 0.0
        self._start = None
        self._evals_mark = 0
        self._battle_mark = (0, 0)
        self._table_size_start = len(bs.dataset["damage_dictionary"])
        self._resumed = False
        self.stopped = False

    # ------------------------------------------------------------------ setup

    @classmethod
    def new_run_dir(cls, config: GAConfig) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = RUNS_DIR / f"{stamp}_{config.preset}"
        n = 1
        while path.exists():
            n += 1
            path = RUNS_DIR / f"{stamp}_{config.preset}_{n}"
        return path

    @classmethod
    def resume(cls, run_dir: Path | str, generations: int | None = None, **kwargs) -> "GeneticAlgorithm":
        run_dir = Path(run_dir)
        cfg = GAConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        if generations:
            cfg.generations = generations
        ga = cls(cfg, run_dir=run_dir, **kwargs)
        ckpt = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        ga.generation = ckpt["generation"]
        ga.population = [Individual(ops.normalize(p["genome"]), p["origin"], p["born"]) for p in ckpt["population"]]
        state = ckpt["rng_state"]
        ga.rng.setstate((state[0], tuple(state[1]), state[2]))
        ga.best_history = ckpt["best_history"]
        ga.mode_history = ckpt["mode_history"]
        ga.mode = ckpt.get("mode")
        ga.best_found_gen = ckpt["best_found_gen"]
        ga.improvements = ckpt.get("improvements", [])
        ga.elapsed_before = ckpt.get("elapsed", 0.0)
        hof_path = run_dir / "hall_of_fame.json"
        if hof_path.exists():
            for entry in json.loads(hof_path.read_text(encoding="utf-8")):
                ga.hall_of_fame[ops.genome_key(entry["genome"])] = entry
        # Drop history lines after the checkpoint (they will be recomputed)
        hist = run_dir / "history.jsonl"
        if hist.exists():
            lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and json.loads(ln)["gen"] <= ga.generation]
            hist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        ga._resumed = True
        return ga

    # ------------------------------------------------------------------ evaluation

    def _evaluate(self, individuals: list) -> None:
        todo = [ind for ind in individuals if ind.ev is None]
        if hasattr(self.evaluator, "prefetch"):  # random battles: missing battles in vectorised batches
            map_chunks = (lambda chunks: self.pool.map(_simulate_items, chunks)) if self.pool is not None else None
            self.evaluator.prefetch([ind.genome for ind in todo], map_chunks=map_chunks,
                                    n_chunks=self.cfg.workers * 2)
            if self.should_stop():
                raise StopRequested
            for ind in todo:
                ind.ev = self.evaluator.evaluate(ind.genome)
            return
        if self.pool is not None and len(todo) > 20:
            pending = {}
            for ind in todo:
                cached = self.evaluator._genomes.get(ind.key)
                if cached is not None:
                    ind.ev = cached
                else:
                    pending.setdefault(ind.key, []).append(ind)
            keys = list(pending)
            genomes = [pending[k][0].genome for k in keys]
            n = max(1, len(genomes) // (self.cfg.workers * 4))
            chunks = [genomes[i:i + n] for i in range(0, len(genomes), n)]
            results = []
            for chunk_result in self.pool.map(_evaluate_chunk, chunks):
                results.extend(chunk_result)
                if self.should_stop():
                    raise StopRequested
            for k, ev in zip(keys, results):
                self.evaluator._genomes[k] = ev
                for ind in pending[k]:
                    ind.ev = ev
            self.evaluator.evaluations += len(todo)
            return
        for i, ind in enumerate(todo):
            ind.ev = self.evaluator.evaluate(ind.genome)
            if i % 50 == 49 and self.should_stop():
                raise StopRequested

    # ------------------------------------------------------------------ operators

    def _new_random(self, n: int, origin: str, born: int) -> list:
        cfg = self.cfg
        return [Individual(ops.generate_individual(self.rng, smart=cfg.smart_init, pkdex=bs.dataset["pkdex"],
                                                   switching=cfg.switching, plan_prob=cfg.plan_init),
                           origin, born) for _ in range(n)]

    def _select_parents(self, pop: list, n: int) -> list:
        if n <= 0:
            return []
        if self.cfg.selection == "tournament":
            k = self.cfg.tournament_k
            out = []
            for _ in range(n):
                contenders = [pop[self.rng.randrange(len(pop))] for _ in range(k)]
                out.append(min(contenders, key=lambda ind: ind.fitness))
            return out
        # Rank-based roulette without replacement (Efraimidis-Spirakis): the best of N
        # individuals has weight N and the worst weight 1.
        ranked = sorted(pop, key=lambda ind: ind.fitness)
        size = len(ranked)
        keyed = [(self.rng.random() ** (1.0 / (size - i)), ind) for i, ind in enumerate(ranked)]
        keyed.sort(key=lambda t: t[0], reverse=True)
        return [ind for _, ind in keyed[:min(n, size)]]

    def _params(self) -> dict:
        cfg = self.cfg
        params = {"selected_pct": cfg.selected_pct, "cross_pct": cfg.cross_pct, "mut_catches": cfg.mut_catches,
                  "mut_team": cfg.mut_team, "survivors": cfg.survivors}
        if cfg.adaptive_modes and self.mode in ("exploration", "exploitation"):
            m = cfg.exploration if self.mode == "exploration" else cfg.exploitation
            params.update({k: m[k] for k in ("selected_pct", "cross_pct", "mut_catches", "mut_team") if k in m})
            if "survivors_pct" in m:
                params["survivors"] = max(1, int(m["survivors_pct"] * cfg.population))
        return params

    def _update_mode(self, current_best: float) -> None:
        """Exploration / exploitation switching of the original implementation."""
        if not self.cfg.adaptive_modes:
            self.mode = None
            return
        if len(self.best_history) < 5:
            mode = "exploration"
        else:
            last_five = self.best_history[-5:]
            last_modes = self.mode_history[-5:]
            min_exploitation_done = bool(last_modes) and all(m == "exploitation" for m in last_modes)
            if self.mode_history and self.mode_history[-1] == "exploitation" and not min_exploitation_done:
                mode = "exploitation"
            elif current_best < min(last_five):
                mode = "exploitation"
            else:
                mode = "exploration"
        self.mode = mode

    def _unique_sorted(self, individuals: list) -> list:
        seen = set()
        out = []
        for ind in sorted(individuals, key=lambda i: i.fitness):
            if ind.key in seen:
                continue
            seen.add(ind.key)
            out.append(ind)
        return out

    def _step(self, g: int) -> list:
        cfg = self.cfg
        params = self._params()
        n = cfg.population
        pop = self.population

        parents = self._select_parents(pop, int(round(params["selected_pct"] * n)))
        self.rng.shuffle(parents)
        n_cross = int(round(params["cross_pct"] * len(parents)))
        n_cross -= n_cross % 2
        offspring = []
        for a, b in zip(parents[0:n_cross:2], parents[1:n_cross:2]):
            for _ in range(cfg.children_per_pair):
                offspring.append(Individual(ops.crossover(a.genome, b.genome, self.rng), "crossover", g))
        for p in parents[n_cross:]:
            child = ops.mutate(p.genome, self.rng, params["mut_catches"], params["mut_team"], cfg.mut_order,
                               cfg.mut_plan if cfg.switching else 0.0)
            offspring.append(Individual(child, "mutation", g))
        self._evaluate(offspring)

        survivors = self._unique_sorted(pop + offspring)[:params["survivors"]]
        nxt = list(survivors)
        if cfg.fill_strategy == "random":
            fresh = self._new_random(n - len(nxt), "immigrant", g)
            self._evaluate(fresh)
            nxt += fresh
        else:
            n_imm = int(round(cfg.immigrants_pct * n))
            taken = {ind.key for ind in nxt}
            for ind in self._unique_sorted(offspring):
                if len(nxt) >= n - n_imm:
                    break
                if ind.key not in taken:
                    nxt.append(ind)
                    taken.add(ind.key)
            for ind in self._unique_sorted(pop):  # not enough distinct children
                if len(nxt) >= n - n_imm:
                    break
                if ind.key not in taken:
                    nxt.append(ind)
                    taken.add(ind.key)
            fresh = self._new_random(n - len(nxt), "immigrant", g)
            self._evaluate(fresh)
            nxt += fresh
        return nxt

    # ------------------------------------------------------------------ bookkeeping

    def _hof_entry(self, ind: Individual, g: int) -> dict:
        ev = ind.ev
        return {"genome": ind.genome, "fitness": _finite(ev.fitness), "wins": ev.wins, "losses": ev.losses,
                "hp_lost": ev.hp_lost, "per_battle": ev.per_battle_values(), "gen_found": g, "origin": ind.origin,
                "p_complete": ev.p_complete}

    def _update_hall_of_fame(self, g: int) -> bool:
        changed = False
        worst = max((e["fitness"] for e in self.hall_of_fame.values()), default=INF) \
            if len(self.hall_of_fame) >= HALL_OF_FAME_SIZE else INF
        for ind in self._unique_sorted(self.population):
            if math.isinf(ind.fitness) or ind.fitness >= worst and len(self.hall_of_fame) >= HALL_OF_FAME_SIZE:
                break
            if ind.key not in self.hall_of_fame:
                self.hall_of_fame[ind.key] = self._hof_entry(ind, g)
                changed = True
        if len(self.hall_of_fame) > HALL_OF_FAME_SIZE:
            ordered = sorted(self.hall_of_fame.items(), key=lambda kv: kv[1]["fitness"])
            self.hall_of_fame = dict(ordered[:HALL_OF_FAME_SIZE])
        return changed

    def hall_of_fame_list(self) -> list:
        return sorted(self.hall_of_fame.values(), key=lambda e: e["fitness"])

    def _diversity(self) -> float:
        pop = self.population
        if len(pop) < 2:
            return 0.0
        total = 0.0
        pairs = 100
        for _ in range(pairs):
            a, b = self.stats_rng.sample(pop, 2)
            total += sum(x != y for x, y in zip(a.genome[0], b.genome[0])) / ops.N_ROUTES
        return total / pairs

    def _stats(self, g: int, gen_seconds: float) -> dict:
        pop = self.population
        fits = sorted(ind.fitness for ind in pop if not math.isinf(ind.fitness))
        best = min(pop, key=lambda ind: ind.fitness)
        improved = self.best is None or best.fitness < self.best.fitness
        if improved:
            if self.best is not None:
                self.improvements.append({"gen": g, "from": _finite(self.best.fitness), "to": _finite(best.fitness),
                                          "origin": best.origin})
            self.best = best
            self.best_found_gen = g
        ev = self.best.ev
        wins_hist = [0] * (ops.N_TRAINERS + 1)
        origin_counts = {o: 0 for o in ORIGINS}
        for ind in pop:
            if ind.ev is not None and ind.ev.per_battle:
                wins_hist[ind.ev.wins] += 1
            origin_counts[ind.display_origin(g)] += 1
        evals = self.evaluator.evaluations
        hits, misses = self.evaluator.battle_hits, self.evaluator.battle_misses
        d_hits, d_miss = hits - self._battle_mark[0], misses - self._battle_mark[1]
        sample = pop if len(pop) <= POPULATION_SAMPLE_UI else self.stats_rng.sample(pop, POPULATION_SAMPLE_UI)
        stats = {
            "gen": g,
            "generations": self.cfg.generations,
            "elapsed": round(self.elapsed_before + (time.time() - self._start), 2),
            "gen_seconds": round(gen_seconds, 3),
            "evaluations": evals,
            "evals_per_s": round((evals - self._evals_mark) / gen_seconds, 1) if gen_seconds > 0 else None,
            "battle_hit_rate": round(d_hits / (d_hits + d_miss), 4) if d_hits + d_miss else None,
            "best": _finite(self.best.fitness),
            "gen_best": _finite(best.fitness),
            "mean": round(statistics.fmean(fits), 2) if fits else None,
            "median": statistics.median(fits) if fits else None,
            "p10": _percentile(fits, 0.10) if fits else None,
            "p90": _percentile(fits, 0.90) if fits else None,
            "worst": fits[-1] if fits else None,
            "n_invalid": len(pop) - len(fits),
            "best_wins": ev.wins,
            "best_losses": ev.losses,
            "best_hp_lost": ev.hp_lost,
            "best_per_battle": ev.per_battle_values(),
            "best_switches": sum(b.switches for b in ev.per_battle),
            "best_p_complete": ev.p_complete,
            "best_win_rates": [b.win_rate for b in ev.per_battle] if ev.p_complete is not None else None,
            "best_origin": self.best.origin,
            "best_found_gen": self.best_found_gen,
            "improved": improved,
            "best_genome": self.best.genome if improved else None,
            "unique_ratio": round(len({ind.key for ind in pop}) / len(pop), 4),
            "diversity": round(self._diversity(), 4),
            "wins_hist": wins_hist,
            "origin_counts": origin_counts,
            "mode": self.mode,
            "params": self._params(),
            "population": [[_finite(ind.fitness), ORIGINS.index(ind.display_origin(g))] for ind in sample],
        }
        self._evals_mark = evals
        self._battle_mark = (hits, misses)
        return stats

    # ------------------------------------------------------------------ persistence

    def _write_json(self, name: str, data) -> None:
        bs.save_json_atomic(data, self.run_dir / name)

    def _save_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self._resumed:
            self._write_json("config.json", self.cfg.to_dict())
            (self.run_dir / "history.jsonl").write_text("", encoding="utf-8")
        else:
            self._write_json("config.json", self.cfg.to_dict())

    def _append_history(self, stats: dict) -> None:
        line = dict(stats)
        pop = line.pop("population")
        line["population"] = pop[:POPULATION_SAMPLE_HISTORY]
        with open(self.run_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def save_checkpoint(self) -> None:
        if not self.run_dir:
            return
        state = self.rng.getstate()
        self._write_json("checkpoint.json", {
            "generation": self.generation,
            "population": [{"genome": ind.genome, "origin": ind.origin, "born": ind.born} for ind in self.population],
            "rng_state": [state[0], list(state[1]), state[2]],
            "best_history": self.best_history,
            "mode_history": self.mode_history,
            "mode": self.mode,
            "best_found_gen": self.best_found_gen,
            "improvements": self.improvements,
            "elapsed": self.elapsed_before + (time.time() - self._start),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        })
        if self.best is not None:
            self._write_json("best.json", {"format": "pkm-nuzlocke-individual/v1",
                                           **self._hof_entry(self.best, self.best_found_gen)})
        if len(bs.dataset["damage_dictionary"]) > self._table_size_start:
            bs.save_damage_dictionary(bs.dataset)
            self._table_size_start = len(bs.dataset["damage_dictionary"])

    # ------------------------------------------------------------------ run

    def _finish_generation(self, g: int, t0: float) -> dict:
        self.generation = g
        stats = self._stats(g, time.time() - t0)
        self._update_mode(self.best.fitness)
        self.best_history.append(self.best.fitness)
        self.mode_history.append(self.mode)
        hof_changed = self._update_hall_of_fame(g)
        stats["hof_changed"] = hof_changed
        if self.run_dir:
            self._append_history(stats)
            if hof_changed:
                self._write_json("hall_of_fame.json", self.hall_of_fame_list())
            if g % self.cfg.checkpoint_every == 0 or g == self.cfg.generations:
                self.save_checkpoint()
        if self.on_generation:
            self.on_generation(stats, self)
        return stats

    def run(self) -> dict:
        self._start = time.time()
        if self.cfg.workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=self.cfg.workers, initializer=_init_worker,
                                            initargs=(self.cfg.to_dict(),))
        if self.run_dir:
            self._save_start()
        try:
            t0 = time.time()
            if not self._resumed:
                self.population = self._new_random(self.cfg.population, "initial", 0)
                self._evaluate(self.population)
                self._finish_generation(0, t0)
            else:
                self._evaluate(self.population)
                self.best = min(self.population, key=lambda ind: ind.fitness)
                self.log(f"Reanudando desde la generación {self.generation} (mejor {_finite(self.best.fitness)})")
            for g in range(self.generation + 1, self.cfg.generations + 1):
                self.wait_if_paused()
                if self.should_stop():
                    raise StopRequested
                t0 = time.time()
                self.population = self._step(g)
                self._finish_generation(g, t0)
        except StopRequested:
            self.stopped = True
        finally:
            if self.pool is not None:
                self.pool.shutdown(cancel_futures=True)
            if self.run_dir and self.population and all(ind.ev is not None for ind in self.population):
                self.save_checkpoint()
                self._write_json("hall_of_fame.json", self.hall_of_fame_list())
        return {
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "generations_done": self.generation,
            "stopped": self.stopped,
            "best": self._hof_entry(self.best, self.best_found_gen) if self.best else None,
            "evaluator": self.evaluator.stats(),
            "elapsed": round(self.elapsed_before + (time.time() - self._start), 1),
        }


def load_individual(data) -> list:
    """Accepts a genome ``[catches, teams]``, an individual JSON (``catches``/``teams``) or a
    hall of fame entry (``genome``) and returns the genome."""
    if isinstance(data, dict):
        if "genome" in data:
            return load_individual(data["genome"])
        if "catches" in data and "teams" in data:
            return ops.normalize([data["catches"], data["teams"], data.get("plans")])
        raise ValueError("JSON sin 'genome' ni 'catches'/'teams'")
    if isinstance(data, (list, tuple)) and len(data) in (2, 3):
        catches, teams = data[0], data[1]
        if len(catches) == ops.N_ROUTES and len(teams) == ops.N_TRAINERS:
            return ops.normalize(data)
    raise ValueError(f"Se esperaba [capturas ({ops.N_ROUTES}), equipos ({ops.N_TRAINERS}), plan de cambios opcional]")


def individual_to_json(genome: list, evaluation: Evaluation | None = None, **meta) -> dict:
    genome = ops.normalize(genome)
    out = {"format": "pkm-nuzlocke-individual/v2", "catches": genome[0], "teams": genome[1], "plans": genome[2]}
    if evaluation is not None:
        out.update({"fitness": _finite(evaluation.fitness), "raw_fitness": _finite(evaluation.raw_fitness),
                    "wins": evaluation.wins, "losses": evaluation.losses, "hp_lost": evaluation.hp_lost,
                    "per_battle": evaluation.per_battle_values(), "violations": evaluation.violations})
    if meta:
        out["meta"] = meta
    return out


def list_runs(runs_dir: Path = RUNS_DIR) -> list:
    """Summary of the saved runs (most recent first)."""
    out = []
    if not runs_dir.exists():
        return out
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        last = None
        hist = d / "history.jsonl"
        if hist.exists():
            with open(hist, "rb") as fh:
                try:
                    fh.seek(-200_000, os.SEEK_END)
                except OSError:
                    fh.seek(0)
                lines = [ln for ln in fh.read().decode("utf-8", "ignore").splitlines() if ln.startswith("{")]
            for ln in reversed(lines):
                try:
                    last = json.loads(ln)
                    break
                except json.JSONDecodeError:
                    continue
        out.append({"path": str(d), "name": d.name, "config": cfg, "last": last,
                    "resumable": (d / "checkpoint.json").exists()})
    return out


def read_history(run_dir: Path | str) -> list:
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out
