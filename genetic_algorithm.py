"""Command line runner for the genetic algorithm.

Examples:
    python genetic_algorithm.py --preset memoria --generations 300 --seed 1
    python genetic_algorithm.py --preset mejorado --population 1000 --workers 4
    python genetic_algorithm.py --preset mejorado --scenarios 100   (combates aleatorios)
    python genetic_algorithm.py --resume runs/20261114-231500_mejorado --generations 500

Each run is saved in runs/<fecha>_<preset>/ (config, history, hall of fame,
best individual and checkpoint) and can be opened from the interface (python app.py).
"""
import argparse
import sys

from ga.config import PRESETS, preset
from ga.engine import GeneticAlgorithm


def _fmt(value, width=8):
    return f"{value:>{width}.0f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"


def print_generation(stats: dict, _engine=None) -> None:
    mark = " *" if stats["improved"] and stats["gen"] > 0 else ""
    mode = f" | modo {stats['mode']}" if stats.get("mode") else ""
    hit = f"{stats['battle_hit_rate']:.0%}" if stats.get("battle_hit_rate") is not None else "-"
    switches = f" | cambios {stats['best_switches']}" if stats.get("best_switches") else ""
    p_ok = f" | P(completar) {stats['best_p_complete']:.1%}" if stats.get("best_p_complete") is not None else ""
    print(f"Gen {stats['gen']:4d}/{stats['generations']} | mejor {_fmt(stats['best'])} | mediana {_fmt(stats['median'])}"
          f" | victorias {stats['best_wins']:2d}/21{p_ok}{switches} | únicos {stats['unique_ratio']:.0%} | div {stats['diversity']:.2f}"
          f" | {stats['evals_per_s'] or 0:6.0f} eval/s | caché {hit}{mode}{mark}", flush=True)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="memoria")
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, help="procesos para evaluar en paralelo")
    parser.add_argument("--adaptive-modes", action="store_true", help="cambio exploración/explotación")
    switching = parser.add_mutually_exclusive_group()
    switching.add_argument("--switching", dest="switching", action="store_true", default=None,
                           help="los individuos llevan un plan de cambios de Pokémon por combate")
    switching.add_argument("--no-switching", dest="switching", action="store_false",
                           help="sin cambios de Pokémon (como en la memoria)")
    parser.add_argument("--scenarios", type=int, help="combates aleatorios: escenarios por combate (0 = determinista)")
    parser.add_argument("--scenario-seed", type=int, help="semilla de los escenarios aleatorios")
    parser.add_argument("--graded-penalty", action="store_true", help="derrota = 10000 + HP rival restante")
    parser.add_argument("--run-dir", help="carpeta donde guardar la ejecución (por defecto runs/<fecha>_<preset>)")
    parser.add_argument("--no-save", action="store_true", help="no guardar la ejecución en disco")
    parser.add_argument("--resume", metavar="RUN_DIR", help="reanudar una ejecución guardada")
    args = parser.parse_args(argv)

    if args.resume:
        ga = GeneticAlgorithm.resume(args.resume, generations=args.generations, on_generation=print_generation)
    else:
        cfg = preset(args.preset).with_overrides(population=args.population, generations=args.generations,
                                                 seed=args.seed, workers=args.workers)
        if args.adaptive_modes:
            cfg.adaptive_modes = True
        if args.switching is not None:
            cfg.switching = args.switching
        if args.scenarios is not None:
            cfg.scenarios = args.scenarios
        if args.scenario_seed is not None:
            cfg.scenario_seed = args.scenario_seed
        if args.graded_penalty:
            cfg.graded_loss_penalty = True
        if args.population:  # keep the survivors share of the preset
            base = PRESETS[args.preset]
            cfg.survivors = max(1, round(base.survivors * args.population / base.population))
        run_dir = None if args.no_save else (args.run_dir or GeneticAlgorithm.new_run_dir(cfg))
        ga = GeneticAlgorithm(cfg, run_dir=run_dir, on_generation=print_generation)
    print(f"Semilla: {ga.cfg.seed} | Carpeta: {ga.run_dir or '(sin guardar)'}")
    try:
        result = ga.run()
    except KeyboardInterrupt:
        ga.stopped = True
        result = {"best": None, "stopped": True}
    best = result.get("best")
    if best:
        print("-" * 60)
        print(f"Mejor fitness: {best['fitness']} (gen {best['gen_found']}, {best['wins']}/21 victorias, "
              f"{best['hp_lost']} % HP perdido)")
        print(f"Genotipo: {best['genome']}")
    if result.get("run_dir"):
        print(f"Ejecución guardada en {result['run_dir']}")


if __name__ == "__main__":
    main()
