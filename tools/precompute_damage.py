"""Builds the full damage table (pkm_data/damage_dictionary.json).

Requires the damage server running:  node damage_calculator/servidor.js
Usage:  python tools/precompute_damage.py [--check] [--force] [--rolls]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import battle_simulator as bs  # noqa: E402
from ga import damage_table  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="solo muestra la cobertura de la tabla")
    parser.add_argument("--force", action="store_true", help="recalcula también las claves existentes")
    parser.add_argument("--url", default=bs.DAMAGE_SERVER_URL)
    parser.add_argument("--rolls", action="store_true",
                        help="genera también pkm_data/damage_rolls.json (todos los movimientos, para combates aleatorios)")
    args = parser.parse_args()

    present, required = damage_table.coverage(bs.dataset)
    print(f"Tabla de daños: {present}/{required} claves ({present / required:.1%})")
    if args.check:
        return
    if not damage_table.server_available(args.url):
        sys.exit(f"El servidor {args.url} no responde. Arráncalo con: node damage_calculator/servidor.js")

    def progress(done, total):
        print(f"\r  {done}/{total} cálculos", end="", flush=True)

    summary = damage_table.precompute(bs.dataset, url=args.url, force=args.force, progress=progress)
    print()
    print(f"Claves calculadas: {summary['computed_keys']} ({summary['calcs']} cálculos en {summary['seconds']} s)")
    print(f"Tabla guardada con {summary['table_size']} entradas en {bs.DAMAGE_DICTIONARY_PATH}")
    if summary["errors"]:
        print(f"{len(summary['errors'])} cálculos con error (se cuentan como 0 de daño), p. ej.: {summary['errors'][:3]}")
    if summary["collisions"]:
        print(f"AVISO: {len(summary['collisions'])} claves pedidas por entrenadores con distintas MT: {summary['collisions'][:3]}")

    if args.rolls:
        print("Tabla de tiradas (todos los movimientos, normal y crítico) para los combates aleatorios:")
        rolls = damage_table.precompute_rolls(bs.dataset, url=args.url, progress=progress)
        print()
        print(f"{rolls['keys']} claves ({rolls['calcs']} cálculos en {rolls['seconds']} s) guardadas en {rolls['path']}")
        if rolls["errors"]:
            print(f"{len(rolls['errors'])} cálculos con error (0 de daño), p. ej.: {rolls['errors'][:3]}")
        if rolls["roll_mismatches"]:
            print(f"AVISO: {rolls['roll_mismatches']} tiradas no siguen floor(base * r / 100)")


if __name__ == "__main__":
    main()
