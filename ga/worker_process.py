"""Entry point of the GA child process used by the interface (no Qt imports here)."""
from __future__ import annotations

import time
import traceback


def run_ga_process(config: dict | None, run_dir: str | None, resume_dir: str | None, generations: int | None,
                   queue, stop_event, pause_event) -> None:
    try:
        from ga.config import GAConfig
        from ga.engine import GeneticAlgorithm

        def on_generation(stats, ga):
            queue.put(("generation", stats))
            if stats.get("hof_changed"):
                queue.put(("hof", ga.hall_of_fame_list()))

        def wait_if_paused():
            if pause_event.is_set():
                queue.put(("paused", None))
                while pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.1)
                queue.put(("resumed", None))

        kwargs = dict(on_generation=on_generation, should_stop=stop_event.is_set, wait_if_paused=wait_if_paused,
                      log=lambda msg: queue.put(("log", str(msg))))
        if resume_dir:
            ga = GeneticAlgorithm.resume(resume_dir, generations=generations, **kwargs)
            history = []
            from ga.engine import read_history

            history = read_history(resume_dir)
            queue.put(("started", {"run_dir": str(ga.run_dir), "config": ga.cfg.to_dict(), "history": history,
                                   "hall_of_fame": ga.hall_of_fame_list()}))
        else:
            cfg = GAConfig.from_dict(config)
            ga = GeneticAlgorithm(cfg, run_dir=run_dir or GeneticAlgorithm.new_run_dir(cfg), **kwargs)
            queue.put(("started", {"run_dir": str(ga.run_dir), "config": ga.cfg.to_dict(), "history": [],
                                   "hall_of_fame": []}))
        result = ga.run()
        queue.put(("finished", result))
    except Exception:  # reported to the interface
        queue.put(("error", traceback.format_exc()))
