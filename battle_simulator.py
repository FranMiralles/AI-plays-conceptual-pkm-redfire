"""Deterministic battle simulator for the FireRed nuzlocke problem.

In this version the individual only chooses the team (and its send-out order)
used against each trainer. Every turn each active Pokémon uses the move that
maximises the damage dealt to the current rival (looked up in the damage
table), and the active Pokémon cannot switch until it faints.

Damage values come from the @smogon/calc server in ``damage_calculator/``
(median damage roll, no criticals). Results are cached in
``pkm_data/damage_dictionary.json``; run ``python tools/precompute_damage.py``
once to build the full table so the server is no longer needed.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import requests

from route_data.evolve_obj import AVAILABLE_EVOLVE_OBJ_TRAINERS
from route_data.mt_moves import AVAILABLE_MT_TRAINERS
from route_data.trainers import PREVIOUS_ROUTES_TO_TRAINER, TRAINERS, TRAINERS_ORDER

ROOT = Path(__file__).resolve().parent
PKM_DATA = ROOT / "pkm_data"
DAMAGE_DICTIONARY_PATH = PKM_DATA / "damage_dictionary.json"
DAMAGE_SERVER_URL = os.environ.get("PKM_DAMAGE_SERVER", "http://localhost:3000")

# A battle that reaches this many turns (e.g. both sides deal 0 damage) is a loss.
MAX_TURNS = 300
# Penalty added to the fitness for every lost battle (as in the project report).
LOSS_PENALTY = 10000
# Fix over the original simulator: a Pokémon entering the field does not inherit the
# charging (Solar Beam, Dig, Fly, Dive) or recharge state of the one that fainted.
# Set to False to reproduce the original results exactly.
RESET_VOLATILE_ON_SWITCH = True
# Fix over the original simulator: player->rival and rival->player lookups shared the
# same key format, so "Pidgey|9|Charmander|9" could mean the player's Pidgey (player
# moves) or the rival's Pidgey (rival moves) and one side used the other's best move.
# With the fix, rival attacks get their own keys ("...|R"). Set to False to reproduce
# the original results.
SEPARATE_RIVAL_DAMAGE_KEYS = True
DAMAGE_TABLE_FORMAT = "v2-separate-rival-keys" if SEPARATE_RIVAL_DAMAGE_KEYS else "v1"
FORMAT_KEY = "__format__"

DRAIN_MOVES = ("absorb", "mega-drain", "giga-drain")
EXPLOSION_MOVES = ("explosion", "self-destruct")
# Two-turn moves: first turn charges (no damage), second turn hits.
# Dig/Fly/Dive also make the user semi-invulnerable during the charging turn.
CHARGE_MOVES = ("solar-beam", "dig", "fly", "dive")
SEMI_INVULNERABLE_MOVES = ("dig", "fly", "dive")
SIDE_NAMES = ("player", "rival")

# Trainers whose team depends on the player's starter (catches[0]).
RIVAL_TRAINER_INDICES = frozenset(i for i, name in enumerate(TRAINERS_ORDER) if name.startswith("RIVAL"))


class DamageServerError(RuntimeError):
    """Raised when a damage value is missing from the table and the server is unavailable."""


# --------------------------------------------------------------------------- data


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"El archivo {path} no existe.")
        return {}
    except json.JSONDecodeError as exc:
        print(f"Error al decodificar el JSON en {path}: {exc}")
        return {}


def load_damage_dictionary(path: Path = DAMAGE_DICTIONARY_PATH) -> dict:
    """Loads the damage table, discarding it if it was built with another key format."""
    if not Path(path).exists():
        return {FORMAT_KEY: DAMAGE_TABLE_FORMAT}
    table = _load_json(path)
    fmt = table.get(FORMAT_KEY, "v1")
    if fmt != DAMAGE_TABLE_FORMAT:
        print(f"La tabla de daños {path} tiene formato '{fmt}' y se esperaba '{DAMAGE_TABLE_FORMAT}': se ignora. "
              "Regenérala con 'python tools/precompute_damage.py'.")
        return {FORMAT_KEY: DAMAGE_TABLE_FORMAT}
    return table


def load_json_in_dataset() -> dict:
    return {
        "evolutions": _load_json(PKM_DATA / "evolutions.json"),
        "moves": _load_json(PKM_DATA / "moves_priority.json"),
        "pkdex": _load_json(PKM_DATA / "pkdex.json"),
        "damage_dictionary": load_damage_dictionary(),
    }


def save_json_atomic(data, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, path)


def save_damage_dictionary(dataset: dict, path: Path = DAMAGE_DICTIONARY_PATH) -> int:
    """Merge the in-memory damage table with the one on disk and save it. Returns the entry count."""
    on_disk = load_damage_dictionary(path)
    on_disk.update(dataset["damage_dictionary"])
    on_disk[FORMAT_KEY] = DAMAGE_TABLE_FORMAT
    dataset["damage_dictionary"] = on_disk
    save_json_atomic(on_disk, path)
    return len(on_disk) - 1


# --------------------------------------------------------------------------- damage


_session = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def lost_hp_fraction(damage, max_hp: int) -> float:
    """Median damage roll as a fraction of the defender's max HP (2 decimals)."""
    if isinstance(damage, (int, float)):
        return round(damage / max_hp, 2)
    return round(statistics.median(damage) / max_hp, 2)


def deal_damage(attacker: dict, defender: dict, move: str) -> float:
    """Damage dealt by ``attacker`` to ``defender`` with ``move`` as a fraction of HP lost."""
    payload = {"genReq": 3, "attacker": attacker, "defender": defender, "move": move}
    try:
        res = _get_session().post(f"{DAMAGE_SERVER_URL}/calc", json=payload, timeout=10)
    except requests.RequestException as exc:
        raise DamageServerError(
            f"No hay valor de daño precalculado y el servidor {DAMAGE_SERVER_URL} no responde. "
            "Arranca 'node damage_calculator/servidor.js' o ejecuta 'python tools/precompute_damage.py'."
        ) from exc
    data = res.json()
    if "error" in data:
        raise DamageServerError(f"Error del servidor de daño ({attacker['species']} -> {defender['species']}, {move}): {data['error']}")
    return lost_hp_fraction(data["damage"], data["defender"]["rawStats"]["hp"])


def select_best_attack(attacker: dict, target: dict):
    """Returns ``(move, lost_hp_fraction)`` for the attacker's most damaging move against ``target``."""
    best_move = ("", float("-inf"))
    if len(attacker["moves"]) == 0:
        attacker["moves"] = ["tackle"]
    for move in attacker["moves"]:
        lost = deal_damage(
            {"species": attacker["species"], "ability": attacker["ability"], "level": attacker["level"]},
            target,
            move=move,
        )
        if lost > best_move[1]:
            best_move = (move, lost)
    return best_move


def damage_key(attacker_species: str, attacker_level: int, defender_species: str, defender_level: int,
               rival_attacking: bool = False) -> str:
    key = f"{attacker_species}|{attacker_level}|{defender_species}|{defender_level}"
    if rival_attacking and SEPARATE_RIVAL_DAMAGE_KEYS:
        key += "|R"
    return key


# --------------------------------------------------------------------------- helpers


def calculate_speed(base: int, level: int) -> int:
    return int((((2 * base) + 31) * level) / 100 + 5)


def select_level_cap(rival_team: list) -> int:
    """Max level of the rival team; the player's team is levelled to it."""
    return max([1] + [pkm["level"] for pkm in rival_team])


def get_rival_team(trainer_index: int, starter_id) -> list:
    trainer = TRAINERS[TRAINERS_ORDER[trainer_index]]
    if trainer_index in RIVAL_TRAINER_INDICES:
        return trainer[str(starter_id)]
    return trainer


def evolve_pokemon(pkm_id, level_cap: int, available_items: list, dataset: dict) -> int:
    pkm_id = str(pkm_id)
    evolutions = dataset["evolutions"]
    while pkm_id in evolutions:
        target, method, value = evolutions[pkm_id][0], evolutions[pkm_id][1], evolutions[pkm_id][2]
        if ((method == "level-up" and value <= level_cap) or (method == "use-item" and value in available_items)) and target <= 151:
            pkm_id = str(target)
        else:
            break
    return int(pkm_id)


def player_moves(pkm_id: int, level_cap: int, available_mt: list, dataset: dict) -> list:
    return [
        move
        for move, info in dataset["pkdex"][str(pkm_id)]["moves"].items()
        if move not in EXPLOSION_MOVES
        and ((info["method"] == "level-up" and info["level"] <= level_cap) or (info["method"] == "machine" and move in available_mt))
    ]


# --------------------------------------------------------------------------- battle


@dataclass
class BattleResult:
    won: bool
    hp_lost: int  # % HP lost by the player's team (sum over its Pokémon)
    rival_hp_left: int  # % HP the rival team still had when the battle ended
    turns: int
    timed_out: bool
    level_cap: int
    team: list  # evolved player team (pkm ids, send-out order)
    rival_team: list  # rival team as defined in route_data.trainers
    switches: int = 0  # turns the player spent switching Pokémon
    events: list = field(default_factory=list)  # filled when record_events=True

    @property
    def logs(self) -> list:
        return format_events(self)


class _Side:
    __slots__ = ("name", "must_recharge", "charge_counters", "active_index")

    def __init__(self, name):
        self.name = name
        self.must_recharge = False
        self.charge_counters = {m: 0 for m in CHARGE_MOVES}
        self.active_index = None

    def reset_volatile(self):
        self.must_recharge = False
        for m in CHARGE_MOVES:
            self.charge_counters[m] = 0

    def charging(self) -> bool:
        return any(v == 1 for v in self.charge_counters.values())

    def semi_invulnerable_until_hit(self) -> bool:
        """True during the turn the move hits (counter == 2): the user comes back only when attacking."""
        return any(self.charge_counters[m] == 2 for m in SEMI_INVULNERABLE_MOVES)


def _action_kind(side: "_Side") -> str:
    if side.must_recharge:
        return "recharge"
    for move in CHARGE_MOVES:
        if side.charge_counters[move] == 1:
            return move  # charging turn
    return "attack"


def resolve_plan(plan, base_team: list, rival_size: int) -> list:
    """Switch plan as team positions: for each rival Pokémon, the position of the team member
    that must face it, or ``None`` to keep whoever is on the field."""
    out = [None] * rival_size
    if not plan:
        return out
    pos = {pkm: i for i, pkm in enumerate(base_team)}
    for j in range(min(rival_size, len(plan))):
        if plan[j] is not None:
            out[j] = pos.get(plan[j])
    return out


def simulate_battle(team: list, rival_team: list, available_mt: list, available_ev_obj: list, dataset: dict,
                    verbose: bool = False, record_events: bool = False, plan: list | None = None) -> BattleResult:
    """Simulates a battle.

    ``team``: player pkm ids in send-out order (the first one leads).
    ``plan``: optional switch plan, one entry per rival Pokémon with the team member (pkm id)
    that must face it. When that rival Pokémon enters the field and the planned member is alive
    and not on the field, the player spends the turn switching (switching goes first and the
    rival attacks the incoming Pokémon). When the active Pokémon faints, the replacement is free:
    the planned member for the current rival if it is alive, otherwise the next alive Pokémon in
    team order. Without a plan the battle is exactly the original one (no switching).
    """
    level_cap = select_level_cap(rival_team)
    pkdex = dataset["pkdex"]
    priorities = dataset["moves"]
    table = dataset["damage_dictionary"]

    base_team = list(team)
    team = [evolve_pokemon(pkm_id, level_cap, available_ev_obj, dataset) for pkm_id in team]
    n, m = len(team), len(rival_team)
    p_hps = [100] * n
    r_hps = [100] * m
    p_idx, r_idx = 0, 0
    planned = resolve_plan(plan, base_team, m)
    turn = 0
    switches = 0
    timed_out = False
    events = [] if (record_events or verbose) else None
    player, rival = _Side("player"), _Side("rival")
    player.active_index, rival.active_index = 0, 0
    sides = (player, rival)
    hps = (p_hps, r_hps)
    p_alive = n
    p_mons = [None] * n
    matchups = {}  # (p_idx, r_idx) -> (player attack, rival attack, player moves first)

    def planned_switch():
        target = planned[r_idx] if r_idx < m else None
        return target if target is not None and target != p_idx and p_hps[target] > 0 else None

    def replacement():
        target = planned[r_idx] if r_idx < m else None
        if target is not None and p_hps[target] > 0:
            return target
        return next((i for i in range(n) if p_hps[i] > 0), None)

    def player_mon(i: int) -> dict:
        mon = p_mons[i]
        if mon is None:
            entry = pkdex[str(team[i])]
            mon = p_mons[i] = {"id": team[i], "species": entry["name"], "level": level_cap,
                               "ability": entry["abilities"], "speed": calculate_speed(entry["speed"], level_cap),
                               "moves": None}  # computed lazily, only needed when the damage table misses
        return mon

    def best_attack(attacker: dict, defender: dict, rival_attacking: bool):
        key = damage_key(attacker["species"], attacker["level"], defender["species"], defender["level"], rival_attacking)
        attack = table.get(key)
        if attack is None:
            attacker = dict(attacker)
            if attacker.get("moves") is None:
                attacker["moves"] = player_moves(attacker["id"], level_cap, available_mt, dataset)
            attack = select_best_attack(attacker, defender)
            table[key] = attack
        return attack

    def matchup(i: int, j: int):
        mu = matchups.get((i, j))
        if mu is None:
            p_mon, r_mon = player_mon(i), rival_team[j]
            p_attack = best_attack(p_mon, r_mon, False)
            r_attack = best_attack(r_mon, p_mon, True)
            p_prio, r_prio = priorities.get(p_attack[0], 0), priorities.get(r_attack[0], 0)
            first = p_prio > r_prio if p_prio != r_prio else p_mon["speed"] > r_mon["speed"]
            mu = matchups[(i, j)] = (p_attack, r_attack, first)
        return mu

    def record(event: dict):
        r = min(r_idx, m - 1)
        event.update({"p_idx": p_idx, "p_id": team[p_idx], "p_hp": p_hps[p_idx], "r_idx": r,
                      "r_species": rival_team[r]["species"], "r_hp": r_hps[r], "p_hps": list(p_hps),
                      "r_hps": list(r_hps), "dt": n * 100 - sum(p_hps), "dr": m * 100 - sum(r_hps)})
        events.append(event)

    pending = planned_switch()
    while r_idx < m and p_alive > 0:
        turn += 1
        if turn > MAX_TURNS:
            timed_out = True
            break
        # A Pokémon sent out after another fainted does not inherit its charging/recharge state.
        if player.active_index != p_idx:
            if RESET_VOLATILE_ON_SWITCH:
                player.reset_volatile()
            player.active_index = p_idx
        if rival.active_index != r_idx:
            if RESET_VOLATILE_ON_SWITCH:
                rival.reset_volatile()
            rival.active_index = r_idx
        if events is not None:
            record({"type": "turn", "turn": turn})

        player_acts = pending is None
        if not player_acts:  # the player spends the turn switching (switching goes first)
            old, p_idx = p_idx, pending
            pending = None
            switches += 1
            player.reset_volatile()
            player.active_index = p_idx
            if events is not None:
                record({"type": "switch", "turn": turn, "side": "player", "from_idx": old, "to_idx": p_idx,
                        "from_id": team[old], "to_id": team[p_idx]})

        p_attack, r_attack, player_first = matchup(p_idx, r_idx)
        if not player_acts:
            p_attack = None
        attacks = (p_attack, r_attack)
        idx = (p_idx, r_idx)

        uses_hb = (p_attack is not None and p_attack[0] == "hyper-beam" and not player.must_recharge,
                   r_attack[0] == "hyper-beam" and not rival.must_recharge)
        for side, attack in ((player, p_attack), (rival, r_attack)):
            used = attack[0] if attack is not None else None
            counters = side.charge_counters
            for move in CHARGE_MOVES:
                counters[move] = counters[move] + 1 if used == move else 0
        damage = (round(p_attack[1] * 100) if p_attack is not None and not player.must_recharge
                  and not player.charging() else 0,
                  round(r_attack[1] * 100) if not rival.must_recharge and not rival.charging() else 0)

        order = ((0, 1) if player_first else (1, 0)) if player_acts else (1,)
        kinds = [None, None]
        for k in order:
            o = 1 - k
            attack = attacks[k]
            kind = _action_kind(sides[k])
            kinds[k] = kind
            dealt = damage[k]
            # A semi-invulnerable target (Dig, Fly, Dive) cannot be hit
            if kinds[o] is not None:
                if kinds[o] in SEMI_INVULNERABLE_MOVES:
                    dealt = 0
            elif sides[o].semi_invulnerable_until_hit():
                dealt = 0
            target_hps, target = hps[o], idx[o]
            fainted = target_hps[target] <= dealt
            real = target_hps[target] if fainted else dealt
            target_hps[target] -= real
            if fainted and o == 0:
                p_alive -= 1
            if events is not None:
                names = (player_mon(p_idx)["species"], rival_team[r_idx]["species"])
                record({"type": "action", "turn": turn, "side": SIDE_NAMES[k], "attacker": names[k],
                        "defender": names[o], "move": attack[0], "power": attack[1], "kind": kind,
                        "damage": real, "missed": kind == "attack" and attack[1] > 0 and real == 0,
                        "target_hp": target_hps[target], "fainted": fainted})
            if attack[0] in DRAIN_MOVES and real > 0:
                own_hps, own = hps[k], idx[k]
                recovered = min(100 - own_hps[own], real // 2)
                if recovered > 0:
                    own_hps[own] += recovered
                    if events is not None:
                        record({"type": "heal", "turn": turn, "side": SIDE_NAMES[k],
                                "pokemon": (player_mon(p_idx)["species"], rival_team[r_idx]["species"])[k],
                                "amount": recovered})
            if fainted:
                break  # a KO ends the turn: the fainted Pokémon does not act

        # Next turn flags: Hyper Beam must recharge only if the user is still on its feet
        player.must_recharge = uses_hb[0] and p_hps[p_idx] > 0
        rival.must_recharge = uses_hb[1] and r_hps[r_idx] > 0
        for side in sides:
            counters = side.charge_counters
            for move in CHARGE_MOVES:
                if counters[move] > 1:
                    counters[move] = 0

        if r_hps[r_idx] == 0:
            r_idx += 1
            if r_idx < m:
                if events is not None:
                    record({"type": "send_out", "turn": turn, "side": "rival", "idx": r_idx})
                pending = planned_switch()
        elif p_hps[p_idx] == 0:
            nxt = replacement()
            if nxt is not None:
                p_idx = nxt
                if events is not None:
                    record({"type": "send_out", "turn": turn, "side": "player", "idx": p_idx})

    won = (not timed_out) and any(hp > 0 for hp in p_hps)
    result = BattleResult(
        won=won,
        hp_lost=n * 100 - sum(p_hps),
        rival_hp_left=sum(r_hps),
        turns=min(turn, MAX_TURNS),
        timed_out=timed_out,
        level_cap=level_cap,
        team=team,
        rival_team=rival_team,
        switches=switches,
        events=events if events is not None else [],
    )
    if events is not None:
        record({"type": "end", "turn": result.turns, "won": won, "timed_out": timed_out, "hp_lost": result.hp_lost})
    if verbose:
        for line in format_events(result):
            print(line)
    return result


def format_events(result: BattleResult) -> list:
    """Human-readable log lines for a battle simulated with ``record_events=True``."""
    lines = [str(result.team)]
    for ev in result.events:
        if ev["type"] == "turn":
            lines.append(f"TURNO {ev['turn']}")
            lines.append(f"PLAYER {ev['p_id']} HP {ev['p_hp']} | RIVAL {ev['r_species']} HP {ev['r_hp']}")
        elif ev["type"] == "switch":
            lines.append(f"PLAYER switches {ev['from_id']} -> {ev['to_id']}")
        elif ev["type"] == "send_out":
            who = "PLAYER" if ev["side"] == "player" else "RIVAL"
            lines.append(f"{who} sends out {ev['p_id'] if ev['side'] == 'player' else ev['r_species']}")
        elif ev["type"] == "action":
            who = "PLAYER" if ev["side"] == "player" else "RIVAL"
            if ev["kind"] == "recharge":
                lines.append(f"{who} {ev['attacker']} must recharge!")
            elif ev["kind"] != "attack":
                lines.append(f"{who} {ev['attacker']} is charging {ev['kind']}!")
            else:
                lines.append(f"{who} {ev['attacker']} attacks with [{ev['move']}, {ev['power']}] -> {ev['damage']}%")
        elif ev["type"] == "heal":
            lines.append(f"{ev['pokemon']} recovers {ev['amount']} HP")
        elif ev["type"] == "end":
            lines.append("VICTORY" if ev["won"] else ("TIMEOUT" if ev["timed_out"] else "DEFEAT"))
    return lines


# --------------------------------------------------------------------------- fitness


def battle_for_trainer(individual: list, trainer_index: int, dataset: dict, record_events: bool = False,
                       verbose: bool = False) -> BattleResult:
    """Battle ``trainer_index`` of an individual ``[catches, teams]`` or ``[catches, teams, plans]``."""
    catches, teams = individual[0], individual[1]
    plans = individual[2] if len(individual) > 2 else None
    name = TRAINERS_ORDER[trainer_index]
    return simulate_battle(
        team=teams[trainer_index],
        rival_team=get_rival_team(trainer_index, catches[0]),
        available_mt=AVAILABLE_MT_TRAINERS[name],
        available_ev_obj=AVAILABLE_EVOLVE_OBJ_TRAINERS[name],
        dataset=dataset,
        verbose=verbose,
        record_events=record_events,
        plan=plans[trainer_index] if plans else None,
    )


def teams_use_available_pokemon(individual: list) -> bool:
    catches, teams = individual[0], individual[1]
    for i, team in enumerate(teams):
        available = catches[:PREVIOUS_ROUTES_TO_TRAINER[TRAINERS_ORDER[i]]]
        if any(pkm not in available for pkm in team):
            return False
    return True


def calculate_fitness(individual: list, dataset: dict, verbose: bool = False):
    """Legacy API: returns ``(wins_all_battles, fitness, logs_per_battle)``.

    The fitness is the total % HP lost in the 21 battles, +10000 for every lost
    battle; +inf if a team uses a Pokémon not caught before that battle.
    """
    if not teams_use_available_pokemon(individual):
        return (False, float("inf"), [])
    fitness_value = 0
    wins_all = True
    entire_logs = []
    for i in range(len(individual[1])):
        result = battle_for_trainer(individual, i, dataset, record_events=True, verbose=verbose)
        entire_logs.append(result.logs)
        if result.won:
            fitness_value += result.hp_lost
        else:
            wins_all = False
            fitness_value += LOSS_PENALTY
    return (wins_all, fitness_value, entire_logs)


# Shared dataset (loaded once per process)
dataset = load_json_in_dataset()
