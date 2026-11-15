"""Builds route_data/move_data.py: generation 3 (FireRed/LeafGreen) data of every move the
simulator can use: accuracy, critical-hit stage, number of hits, drain/recoil, type, category,
contact and secondary effects (status, confusion, flinch, stat changes, trapping...).

Sources:
  * PokeAPI CSV files (https://github.com/PokeAPI/pokeapi, data/v2/csv): moves.csv has the current
    values and move_changelog.csv the values a move had *before* each change, so the FRLG value is
    the one recorded by the first change after FRLG, if any. move_meta*.csv give the secondary
    effects (current generation), corrected below where generation 3 was different.
  * The damage server (@smogon/calc, generation 3) for type, category and contact.

Usage:  python tools/build_move_data.py <folder with the PokeAPI CSV files> [--url http://localhost:3000]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from route_data.trainers import TRAINERS  # noqa: E402

FRLG_VERSION_GROUP = 7
# Names used by the project that PokeAPI later renamed
ALIASES = {"vise-grip": "vice-grip"}
# Moves that hurt the user when they miss (generation 3: half the damage they would have dealt)
CRASH_MOVES = ("jump-kick", "high-jump-kick")
# Moves that make the user faint
SELF_KO_MOVES = ("self-destruct", "explosion")
# Two-to-three-turn rampage that confuses the user at the end
LOCK_MOVES = ("thrash", "petal-dance", "outrage")
STAT_NAMES = {"attack": "atk", "defense": "def", "special-attack": "spa", "special-defense": "spd", "speed": "spe",
              "accuracy": "acc", "evasion": "eva"}
AILMENTS = {"paralysis": "par", "burn": "brn", "freeze": "frz", "poison": "psn", "sleep": "slp",
            "confusion": "confusion", "trap": "trap"}

# Generation 3 differences with the current PokeAPI data (secondary effects of damaging moves)
GEN3_OVERRIDES = {
    "acid": {"stat_target": [["def", -1]], "stat_chance": 10},  # later: Special Defense
    "crunch": {"stat_target": [["spd", -1]], "stat_chance": 20},  # later: Defense
    "waterfall": {"flinch": 0},  # flinch added in generation 4
    "poison-fang": {"ailment": "tox", "ailment_chance": 30},  # badly poisons; 50 % from generation 6
    "tri-attack": {"ailment": "tri", "ailment_chance": 20},  # paralysis, burn or freeze (1/3 each)
    "secret-power": {"ailment": "par", "ailment_chance": 30},  # buildings / plain terrain
    "rapid-spin": {"stat_self": [], "self_chance": 0, "special": "rapid-spin"},  # speed boost from gen 8
}
# Moves whose power or effect depends on the battle state
SPECIAL = {"facade": "facade", "revenge": "revenge", "flail": "flail", "reversal": "flail", "magnitude": "magnitude",
           "smelling-salt": "smelling-salt", "sheer-cold": "ohko", "fissure": "ohko", "horn-drill": "ohko",
           "guillotine": "ohko", "flame-wheel": "thaw-user", "sacred-fire": "thaw-user", "low-kick": "low-kick"}


def _rows(folder: Path, name: str) -> list:
    with open(folder / f"{name}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def used_moves() -> set:
    pkdex = json.loads((ROOT / "pkm_data" / "pkdex.json").read_text(encoding="utf-8"))
    moves = {m for entry in pkdex.values() for m in entry["moves"]}

    def walk(x):
        if isinstance(x, dict):
            if "species" in x and "moves" in x:
                moves.update(x["moves"])
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(TRAINERS)
    moves.add("tackle")
    return moves


def build(folder: Path, url: str) -> dict:
    moves = {r["identifier"]: r for r in _rows(folder, "moves")}
    by_id = {r["id"]: r["identifier"] for r in moves.values()}
    changes: dict[str, list] = {}
    for r in _rows(folder, "move_changelog"):
        changes.setdefault(by_id.get(r["move_id"]), []).append(r)
    meta = {by_id.get(r["move_id"]): r for r in _rows(folder, "move_meta")}
    ailments = {r["id"]: r["identifier"] for r in _rows(folder, "move_meta_ailments")}
    categories = {r["id"]: r["identifier"] for r in _rows(folder, "move_meta_categories")}
    stats = {r["id"]: r["identifier"] for r in _rows(folder, "stats")}
    stat_changes: dict[str, list] = {}
    for r in _rows(folder, "move_meta_stat_changes"):
        stat_changes.setdefault(by_id.get(r["move_id"]), []).append([STAT_NAMES[stats[r["stat_id"]]], int(r["change"])])

    names = sorted(used_moves())
    calc = requests.post(f"{url}/gen_data", json={"genReq": 3, "moves": names}, timeout=60).json()["moves"]

    out = {}
    for name in names:
        api_name = ALIASES.get(name, name)
        row = moves[api_name]
        later = sorted((c for c in changes.get(api_name, []) if int(c["changed_in_version_group_id"]) > FRLG_VERSION_GROUP),
                       key=lambda c: int(c["changed_in_version_group_id"]))
        accuracy = row["accuracy"]
        for c in later:
            if c["accuracy"]:
                accuracy = c["accuracy"]
                break
        effect_chance = next((c["effect_chance"] for c in later if c["effect_chance"]), None)
        m = meta[api_name]
        category = categories[m["meta_category_id"]]
        ailment = AILMENTS.get(ailments[m["meta_ailment_id"]], "none")
        ailment_chance = int(m["ailment_chance"] or 0)
        changes_list = stat_changes.get(api_name, [])
        stat_chance = int(m["stat_chance"] or 0)
        if effect_chance:  # chance that changed after generation 3
            if ailment != "none":
                ailment_chance = int(effect_chance)
            if stat_chance:
                stat_chance = int(effect_chance)
        entry = {
            "accuracy": int(accuracy) if accuracy else None,  # None: never misses (Swift, Aerial Ace...)
            "crit_stage": int(m["crit_rate"] or 0),  # 1: high critical-hit ratio (1/8 instead of 1/16)
            "hits": [int(m["min_hits"]), int(m["max_hits"])] if m["min_hits"] else [1, 1],
            "drain": int(m["drain"] or 0),  # % of the damage dealt: > 0 heals the user, < 0 is recoil
            "crash": name in CRASH_MOVES,
            "self_ko": name in SELF_KO_MOVES,
            "bp": calc[name]["bp"],  # 0: depends on the battle (Low Kick, Flail...) or fixed damage
            "type": calc[name]["type"].lower(),
            "category": calc[name]["category"].lower(),
            "contact": calc[name]["contact"],
            "ailment": ailment if ailment_chance else "none",
            "ailment_chance": ailment_chance,
            "flinch": int(m["flinch_chance"] or 0),
            "stat_target": changes_list if category == "damage-lower" else [],
            "stat_chance": stat_chance if category == "damage-lower" else 0,
            "stat_self": changes_list if category == "damage-raise" else [],
            "self_chance": stat_chance if category == "damage-raise" else 0,
            "lock": name in LOCK_MOVES,
            "special": SPECIAL.get(name, ""),
        }
        entry.update(GEN3_OVERRIDES.get(name, {}))
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_folder")
    parser.add_argument("--url", default="http://localhost:3000")
    args = parser.parse_args()
    data = build(Path(args.csv_folder), args.url)
    lines = ['"""Generation 3 move data (generated by tools/build_move_data.py from the PokeAPI CSV files and @smogon/calc).',
             "",
             "accuracy: % (None = never misses) · crit_stage: 1 = high critical-hit ratio · hits: [min, max]",
             "drain: % of the damage dealt (> 0 heals, < 0 recoil) · crash: hurts the user on a miss · self_ko: the user faints",
             "bp (base power), type, category (physical/special by type in generation 3), contact",
             "ailment (par, brn, frz, psn, tox, slp, confusion, trap, tri) with ailment_chance %; flinch %",
             "stat_target / stat_self: [[stat, change]] with stat_chance / self_chance %; lock: rampage (Thrash)",
             'special: power or effect that depends on the battle state."""',
             "",
             "MOVE_DATA = {"]
    for name, d in data.items():
        lines.append(f'    "{name}": {json.dumps(d, ensure_ascii=False).replace("true", "True").replace("false", "False").replace("null", "None")},')
    lines.append("}")
    target = ROOT / "route_data" / "move_data.py"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(data)} movimientos escritos en {target}")


if __name__ == "__main__":
    main()
