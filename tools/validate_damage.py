"""Checks the damage formula of the random battle simulator against @smogon/calc (generation 3)
with random stat stages, burns and critical hits.

Requires the damage server running:  node damage_calculator/servidor.js
Usage:  python tools/validate_damage.py [--samples 3000] [--seed 1]
"""
import argparse
import random
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import battle_simulator as bs  # noqa: E402
import stochastic_battle as sb  # noqa: E402
from ga.damage_table import calc_ability_name, load_rolls  # noqa: E402
from route_data.move_data import MOVE_DATA  # noqa: E402


def engine_damage(store, key_id, k, att_stage, def_stage, burned, crit, guts):
    """100 % roll as computed by the engine (same steps as stochastic_battle._simulate_chunk)."""
    phys = bool(store["flags"][key_id, k] & sb.F_PHYS)
    at = int(store["at"][key_id, k])
    if guts and burned and phys:
        at = at * 3 // 2
    sa = max(att_stage, 0) if crit else att_stage
    at = int(sb._stage(at, sa))
    sd = min(def_stage, 0) if crit else def_stage
    df = max(1, int(sb._stage(int(store["df"][key_id, k]), sd)))
    stab = bool(store["flags"][key_id, k] & sb.F_STAB)
    return sb.gen3_damage(int(store["level"][key_id]), at, int(store["bp"][key_id, k]), df, phys, stab,
                          int(store["eff4"][key_id, k]), crit=crit, burned=burned and not guts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--url", default=bs.DAMAGE_SERVER_URL)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    store = sb.default_store()
    rolls = load_rolls()
    species = rolls["__species__"]
    rival_ab = sb._rival_abilities()
    keys = list(store.key_ids.items())
    rng = random.Random(args.seed)
    items, cases = [], []
    while len(items) < args.samples:
        key, key_id = rng.choice(keys)
        k = rng.randrange(sb.K)
        mid = store["move_id"][key_id, k]
        flags = store["flags"][key_id, k]
        if mid < 0 or flags & (sb.F_TABLE | sb.F_FIXED | sb.F_HALF_HP | sb.F_ENDEAVOR | sb.F_OHKO):
            continue
        move = store.move_names[mid]
        if MOVE_DATA[move]["special"] in ("magnitude", "flail"):
            continue
        p = key.split("|")
        att, att_lvl, dfn, def_lvl, rival_attacking = p[0], int(p[1]), p[2], int(p[3]), len(p) > 4
        if rival_attacking:
            att_ab, def_ab = calc_ability_name(rival_ab.get((att, att_lvl), "")), species[dfn]["ability"]
        else:
            att_ab, def_ab = species[att]["ability"], calc_ability_name(rival_ab.get((dfn, def_lvl), ""))
        phys = bool(flags & sb.F_PHYS)
        a_stage, d_stage = rng.randint(-6, 6), rng.randint(-6, 6)
        burned, crit = rng.random() < 0.3, rng.random() < 0.3 and store["crit_thr"][key_id, k] > 0
        guts = sb._norm(att_ab) == "guts"
        attacker = {"species": att, "level": att_lvl, "ability": att_ab,
                    "boosts": {"atk" if phys else "spa": a_stage}, "status": "brn" if burned else ""}
        defender = {"species": dfn, "level": def_lvl, "ability": def_ab, "boosts": {"def" if phys else "spd": d_stage}}
        opts = {"isCrit": True} if crit else {}
        if MOVE_DATA[move]["hits"] != [1, 1] and MOVE_DATA[move]["hits"] != [2, 2]:
            opts["hits"] = 1
        items.append({"attacker": attacker, "defender": defender, "move": move, "moveOptions": opts})
        cases.append((key, move, key_id, k, a_stage, d_stage, burned, crit, guts))
    res = requests.post(f"{args.url}/calc_batch", json={"genReq": 3, "items": items}, timeout=600).json()
    bad = []
    for (key, move, key_id, k, a, d, burned, crit, guts), r in zip(cases, res):
        dmg = r.get("damage", 0)
        if isinstance(dmg, list) and dmg and isinstance(dmg[0], list):
            dmg = dmg[0]
        calc = int(dmg[-1]) if isinstance(dmg, list) and dmg and max(dmg) > 0 else 0
        if MOVE_DATA[move]["hits"] == [2, 2]:
            calc //= 2
        mine = engine_damage(store, key_id, k, a, d, burned, crit, guts)
        if mine != calc:
            bad.append((key, move, a, d, burned, crit, calc, mine))
    print(f"{len(cases) - len(bad)}/{len(cases)} daños idénticos a @smogon/calc con fases, quemadura y críticos")
    for b in bad[:10]:
        print("  distinto:", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
