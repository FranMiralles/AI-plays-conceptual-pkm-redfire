"""Precomputation and coverage of the damage table used by the battle simulator.

The simulator looks up ``attacker|level|defender|level -> [best_move, lost_hp]``.
This module enumerates every key a battle can query (any catchable Pokémon,
evolved at each trainer's level cap, against every rival Pokémon and vice
versa) so the whole table can be built in one go with the damage server.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

import battle_simulator as bs
from route_data.evolve_obj import AVAILABLE_EVOLVE_OBJ_TRAINERS
from route_data.mt_moves import AVAILABLE_MT_TRAINERS
from route_data.routes import ROUTES, ROUTES_ORDER
from route_data.trainers import PREVIOUS_ROUTES_TO_TRAINER, TRAINERS, TRAINERS_ORDER

STARTERS = ("1", "4", "7")


@dataclass
class DamageTask:
    key: str
    attacker: dict  # species, level, ability
    defender: dict  # species, level, ability
    moves: tuple
    trainer: str


def _rival_variants(trainer_index: int):
    trainer = TRAINERS[TRAINERS_ORDER[trainer_index]]
    if isinstance(trainer, dict):
        return [trainer[s] for s in STARTERS]
    return [trainer]


def required_tasks(dataset: dict) -> tuple[list, list]:
    """All damage lookups any individual can trigger.

    Returns ``(tasks, collisions)``; a collision is a key requested by two
    trainers with different move sets (it would make the table ambiguous).
    """
    pkdex = dataset["pkdex"]
    tasks: dict[str, DamageTask] = {}
    collisions = []
    for t_idx, t_name in enumerate(TRAINERS_ORDER):
        cut = PREVIOUS_ROUTES_TO_TRAINER[t_name]
        catchable = {p for route in ROUTES_ORDER[:cut] for p in ROUTES[route]}
        for rival_team in _rival_variants(t_idx):
            cap = bs.select_level_cap(rival_team)
            player_species = {bs.evolve_pokemon(p, cap, AVAILABLE_EVOLVE_OBJ_TRAINERS[t_name], dataset) for p in catchable}
            for p_id in sorted(player_species):
                entry = pkdex[str(p_id)]
                moves = tuple(bs.player_moves(p_id, cap, AVAILABLE_MT_TRAINERS[t_name], dataset)) or ("tackle",)
                p_mon = {"species": entry["name"], "level": cap, "ability": entry["abilities"]}
                for r in rival_team:
                    r_mon = {"species": r["species"], "level": r["level"], "ability": r["ability"]}
                    directions = ((p_mon, r_mon, moves, False), (r_mon, p_mon, tuple(r["moves"]) or ("tackle",), True))
                    for attacker, defender, mv, rival_attacking in directions:
                        key = bs.damage_key(attacker["species"], attacker["level"], defender["species"], defender["level"],
                                            rival_attacking)
                        prev = tasks.get(key)
                        if prev is None:
                            tasks[key] = DamageTask(key, attacker, defender, mv, t_name)
                        elif prev.moves != mv or prev.attacker != attacker or prev.defender != defender:
                            collisions.append((key, prev.trainer, t_name))
    return list(tasks.values()), collisions


def coverage(dataset: dict) -> tuple[int, int]:
    """``(present, required)`` number of keys of the damage table."""
    tasks, _ = required_tasks(dataset)
    table = dataset["damage_dictionary"]
    return sum(1 for t in tasks if t.key in table), len(tasks)


def server_available(url: str = bs.DAMAGE_SERVER_URL, timeout: float = 1.0) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=timeout).ok
    except requests.RequestException:
        return False


def precompute(dataset: dict, url: str = bs.DAMAGE_SERVER_URL, force: bool = False, chunk: int = 2000,
               progress=None) -> dict:
    """Computes the missing keys with the server's ``/calc_batch`` endpoint and saves the table.

    ``progress(done, total)`` is called after every request.
    """
    tasks, collisions = required_tasks(dataset)
    table = dataset["damage_dictionary"]
    todo = [t for t in tasks if force or t.key not in table]
    items, owners = [], []
    for t in todo:
        for move in t.moves:
            items.append({"attacker": t.attacker, "defender": t.defender, "move": move})
            owners.append((t, move))
    results = [None] * len(items)
    errors = []
    session = requests.Session()
    start = time.time()
    for i in range(0, len(items), chunk):
        batch = items[i:i + chunk]
        res = session.post(f"{url}/calc_batch", json={"genReq": 3, "items": batch}, timeout=600)
        res.raise_for_status()
        for j, out in enumerate(res.json()):
            if "error" in out:
                errors.append((owners[i + j][0].key, owners[i + j][1], out["error"]))
                results[i + j] = 0.0
            else:
                results[i + j] = bs.lost_hp_fraction(out["damage"], out["maxHP"])
        if progress:
            progress(min(i + chunk, len(items)), len(items))
    # Reduce to the best move per key (first strict maximum, as select_best_attack)
    best: dict[str, list] = {}
    for (task, move), lost in zip(owners, results):
        cur = best.get(task.key)
        if cur is None or lost > cur[1]:
            best[task.key] = [move, lost]
    table.update(best)
    total = bs.save_damage_dictionary(dataset)
    return {"computed_keys": len(best), "calcs": len(items), "errors": errors, "collisions": collisions,
            "table_size": total, "seconds": round(time.time() - start, 1)}


# --------------------------------------------------------------------------- rolls (random battles)

ROLLS_FORMAT = "rolls-v2"


def _base_damage(damage):
    """``(base, fixed)``: the 100 % roll and whether the damage is fixed (Seismic Toss, Dragon Rage...).

    Generation 3 rolls are ``max(1, floor(base * r / 100))`` for r = 85..100, so the base is enough
    to rebuild the 16 rolls."""
    if isinstance(damage, (int, float)):
        return int(damage), damage > 0  # 0: the defender is immune
    if isinstance(damage, list) and damage and isinstance(damage[0], list):  # one list per hit
        damage = damage[0]
    if not damage or max(damage) == 0:
        return 0, False
    return int(damage[-1]), False


def rolls_from_base(base: int) -> list:
    return [max(1, base * r // 100) if base > 0 else 0 for r in range(85, 101)]


def calc_ability_name(name: str) -> str:
    """'thick-fat' -> 'Thick Fat': @smogon/calc compares ability names exactly, so lowercase names
    (as in route_data/trainers.py and the Pokédex) were silently ignored (no Levitate, Thick Fat...)."""
    return " ".join(w.capitalize() for w in (name or "").replace("_", "-").split("-"))


def _gen3_ability(name: str, species: dict) -> str:
    """Ability of the player's Pokémon in generation 3 (the Pokédex comes from the current
    PokéAPI: Gengar has Cursed Body there, which did not exist in generation 3)."""
    return species[name]["abilities"]["0"]


def precompute_rolls(dataset: dict, url: str = bs.DAMAGE_SERVER_URL, chunk: int = 4000, progress=None) -> dict:
    """Every move of every matchup (not only the best one) with its normal and critical damage
    for one hit, for the random battle simulator. Saved in ``pkm_data/damage_rolls.json`` as
    ``key -> [defender max HP, [[move, base, crit_base, fixed], ...]]``, plus the generation 3
    data needed to recompute the damage with stat changes and burns (``__species__``: base
    stats, types, weight and ability of every species; ``__types__``: type chart)."""
    from route_data.move_data import MOVE_DATA

    tasks, collisions = required_tasks(dataset)
    session = requests.Session()
    names = sorted({t.attacker["species"] for t in tasks} | {t.defender["species"] for t in tasks})
    gen_data = session.post(f"{url}/gen_data", json={"genReq": 3, "species": names, "moves": []}, timeout=120).json()
    species = gen_data["species"]
    items, owners = [], []
    for t in tasks:
        attacker, defender = dict(t.attacker), dict(t.defender)
        rival_attacking = t.key.endswith("|R")
        player, rival = (defender, attacker) if rival_attacking else (attacker, defender)
        player["ability"] = _gen3_ability(player["species"], species)
        rival["ability"] = calc_ability_name(rival["ability"])
        for move in t.moves:
            one_hit = {"hits": 1} if MOVE_DATA.get(move, {}).get("hits", [1, 1]) != [1, 1] else {}
            for crit in (False, True):
                opts = dict(one_hit, **({"isCrit": True} if crit else {}))
                items.append({"attacker": attacker, "defender": defender, "move": move, "moveOptions": opts})
                owners.append((t, move, crit))
    start = time.time()
    out: dict[str, list] = {}
    errors, mismatches = [], 0
    for i in range(0, len(items), chunk):
        res = session.post(f"{url}/calc_batch", json={"genReq": 3, "items": items[i:i + chunk]}, timeout=600)
        res.raise_for_status()
        for j, r in enumerate(res.json()):
            task, move, crit = owners[i + j]
            entry = out.setdefault(task.key, [None, {}])
            if "error" in r:
                errors.append((task.key, move, r["error"]))
                base, fixed = 0, False
            else:
                entry[0] = r["maxHP"]
                base, fixed = _base_damage(r["damage"])
                if MOVE_DATA.get(move, {}).get("hits") == [2, 2] and not fixed:
                    # Fixed two-hit moves ignore hits=1: the calc returns the sum of both hits
                    base //= 2
                elif not fixed and isinstance(r["damage"], list) and r["damage"] and not isinstance(r["damage"][0], list) \
                        and base > 0 and rolls_from_base(base) != r["damage"]:
                    mismatches += 1
            slot = entry[1].setdefault(move, [move, 0, 0, fixed])
            slot[2 if crit else 1] = base
        if progress:
            progress(min(i + chunk, len(items)), len(items))
    table = {"__format__": ROLLS_FORMAT, "__types__": gen_data["types"],
             "__species__": {n: {"baseStats": sp["baseStats"], "types": [x.lower() for x in sp["types"]],
                                 "weightkg": sp["weightkg"], "ability": sp["abilities"]["0"]} for n, sp in species.items()}}
    for key, (max_hp, moves) in out.items():
        table[key] = [max_hp, list(moves.values())]
    bs.save_json_atomic(table, ROLLS_PATH)
    return {"keys": len(out), "calcs": len(items), "errors": errors, "roll_mismatches": mismatches,
            "collisions": collisions, "seconds": round(time.time() - start, 1), "path": str(ROLLS_PATH)}


ROLLS_PATH = bs.PKM_DATA / "damage_rolls.json"


def load_rolls(path=None) -> dict:
    import json

    path = path or ROLLS_PATH
    if not path.exists():
        return {}
    table = json.loads(path.read_text(encoding="utf-8"))
    return table if table.get("__format__") == ROLLS_FORMAT else {}
