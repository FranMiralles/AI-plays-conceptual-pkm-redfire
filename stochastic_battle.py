"""Random battles evaluated over a fixed set of scenarios (common random numbers).

The deterministic simulator (battle_simulator.py) uses the median damage roll, never misses and
never lands a critical hit, so the most powerful move is always the best one. Here every battle is
played once per *scenario*: a scenario is a fixed stream of random numbers that decides, turn by
turn, everything that is random in a generation 3 battle. The fitness of a battle is its mean
cost over the scenarios.

Scenarios are generated from a seed with a counter-based hash (splitmix64), so they are the same
on every machine and run, and scenario ``s`` of a set of 1000 is the same as scenario ``s`` of a
set of 100 (sets are nested). The random numbers are indexed by (scenario, trainer, turn, side,
use), not consumed sequentially: two teams that differ only in the last Pokémon see the same
accuracy checks and rolls in the first turns, which is what makes comparisons between individuals
precise (common random numbers) and the evaluation deterministic and cacheable.

Generation 3 (FireRed/LeafGreen) rules modelled:
  * accuracy (FRLG values) with accuracy/evasion stages and Compound Eyes; Swift, Aerial Ace...
    never miss; one-hit KO moves (30 % + level difference);
  * critical hits (1/16, or 1/8 for Slash, Razor Leaf...), 16 damage rolls per hit, 2-5 hits;
  * the damage is recomputed every hit with the generation 3 formula of @smogon/calc (checked
    against it for every matchup): stat stages (-6..+6; critical hits ignore the unfavourable
    ones), burn (halves physical damage), Guts, Thick Fat, Overgrow/Blaze/Torrent/Swarm at 1/3 HP,
    Facade, SmellingSalt, Revenge, Flail/Reversal, Low Kick, Magnitude, Self-Destruct;
  * secondary effects with their generation 3 chances (route_data/move_data.py): paralysis,
    burn, freeze, poison, bad poison, confusion, flinch, Tri Attack, stat changes on the target
    and on the user (Rock Tomb, Icy Wind, Superpower, Ancient Power...), trapping moves (2-5 turns,
    1/16 HP), Thrash/Petal Dance/Outrage (2-3 turns, then confusion);
  * status conditions: paralysis (speed / 4, 25 % fully paralysed), burn and poison (1/8 HP per
    turn), bad poison (n/16), freeze (20 % thaw per turn, thawed by Fire moves), sleep (2-5
    counter); confusion (2-5 turns, 50 %: 40-power hit to itself); type immunities;
  * abilities: Intimidate, Synchronize, Static, Flame Body, Poison Point, Effect Spore, Shed Skin,
    Natural Cure, Limber, Immunity, Water Veil, Magma Armor, Insomnia, Vital Spirit, Own Tempo,
    Inner Focus, Shield Dust, Clear Body, White Smoke, Hyper Cutter, Keen Eye, Rock Head,
    Compound Eyes, Damp, Early Bird, Sturdy, Serene Grace, Guts, Thick Fat, pinch abilities,
    Battle Armor/Shell Armor and the immunities of Levitate, Flash Fire, Volt/Water Absorb;
  * two-turn moves (Solar Beam, Dig, Fly, Dive, Bounce, Sky Attack, Razor Wind) and the moves that
    hit during them (Thunder/Gust/Twister/Sky Uppercut, Earthquake/Magnitude, Surf), Hyper Beam
    recharge, drain, recoil, crash, Endeavor, Super Fang, False Swipe, Rapid Spin, speed ties.
Not modelled: PP, items, weather, Rollout/Fury Cutter/Rage/Uproar (plain damage), Trace,
Future Sight and moves that need a previous state (Counter, Fake Out, Snore, Spit Up: never used).

Move choice (both sides): if some move KOs for sure with its lowest roll, the most accurate one
(then the one with more priority); otherwise the move with the highest expected damage per turn
(accuracy x mean roll with criticals x mean hits, halved for two-turn and recharge moves). The
choice uses the damage without stat changes, like the simple AI of the game.

The engine is vectorised with NumPy over (battle, scenario) lanes: every lane plays the same turn
at the same time. ``COMPAT_RULES`` + ``MoveStore.from_deterministic`` reproduce the deterministic
simulator exactly (used by the tests to validate the battle logic). ``replay`` records the events
of single battles for the viewer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import battle_simulator as bs
from route_data.evolve_obj import AVAILABLE_EVOLVE_OBJ_TRAINERS
from route_data.move_data import MOVE_DATA
from route_data.trainers import TRAINERS, TRAINERS_ORDER

MAX_TEAM = 6
K = 6  # candidate moves kept per matchup and side
T_CAP = 64  # random numbers per scenario and battle cover 64 turns (longer battles wrap around)
SIDE_SLOTS = 20
SLOTS = 2 * SIDE_SLOTS  # random numbers per turn
N_TRAINERS = len(TRAINERS_ORDER)
HITS_2_5 = np.array([2, 2, 2, 3, 3, 3, 4, 5], dtype=np.int32)  # 3 random bits -> 2..5 hits
MEAN_HITS_2_5 = 3.0
CRIT_THRESHOLD = (4096, 8192)  # out of 65536: 1/16 and 1/8
NEVER_MISS = 101

# random numbers of a side in a turn (uniform 16-bit)
S_ACC, S_CRIT, S_HITS, S_ROLL = 0, 1, 2, 3  # rolls: S_ROLL .. S_ROLL + 4 (one per hit)
S_SEC, S_SEC_KIND, S_DUR, S_PARA, S_THAW, S_CONF, S_CONTACT, S_CONTACT_KIND, S_MAGNITUDE, S_SHED = range(8, 18)

CHARGE_MOVES = frozenset({"solar-beam", "dig", "fly", "dive", "bounce", "sky-attack", "razor-wind"})
SEMI_INVULNERABLE_MOVES = frozenset({"dig", "fly", "dive", "bounce"})
RECHARGE_MOVES = frozenset({"hyper-beam", "blast-burn", "hydro-cannon", "frenzy-plant"})
UNUSABLE_MOVES = frozenset({"fake-out", "snore", "spit-up", "counter", "mirror-coat", "bide", "future-sight"})
SEMI_KIND = {"fly": 1, "bounce": 1, "dig": 2, "dive": 3}  # in the air, underground, underwater
HITS_SEMI = {"thunder": 1, "gust": 1, "twister": 1, "sky-uppercut": 1, "earthquake": 2, "magnitude": 2,
             "surf": 3, "whirlpool": 3}
DOUBLE_SEMI = frozenset({"gust", "twister", "earthquake", "magnitude", "surf", "whirlpool"})

# per (matchup, candidate) flags
(F_CHARGE, F_SEMI, F_RECHARGE, F_CRASH, F_SELF_KO, F_FIXED, F_HALF_HP, F_ENDEAVOR, F_NO_KO, F_TABLE, F_OHKO,
 F_PHYS, F_STAB, F_PINCH, F_CONTACT, F_LOCK) = (1 << i for i in range(16))

# status conditions
NONE, PAR, BRN, FRZ, PSN, TOX, SLP = range(7)
STATUS_CODES = {"par": PAR, "brn": BRN, "frz": FRZ, "psn": PSN, "tox": TOX, "slp": SLP}
STATUS_NAMES = ("", "par", "brn", "frz", "psn", "tox", "slp")
AIL_CONFUSION, AIL_TRI, AIL_TRAP = 7, 8, 9
# stats
ATK, DEF, SPA, SPD, SPE, ACC, EVA = range(7)
STAT_INDEX = {"atk": ATK, "def": DEF, "spa": SPA, "spd": SPD, "spe": SPE, "acc": ACC, "eva": EVA}
STAT_NAMES = ("atk", "def", "spa", "spd", "spe", "acc", "eva")
STAGE_NUM = np.array([2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)  # index stage + 6
STAGE_DEN = np.array([8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
ACC_NUM = np.array([33, 36, 43, 50, 60, 75, 1, 133, 166, 2, 233, 133, 3], dtype=np.int32)
ACC_DEN = np.array([100, 100, 100, 100, 100, 100, 1, 100, 100, 1, 100, 50, 1], dtype=np.int32)
# special moves (per move id)
SP_FACADE, SP_REVENGE, SP_FLAIL, SP_MAGNITUDE, SP_SMELLING_SALT, SP_THAW_USER, SP_RAPID_SPIN = range(1, 8)
SPECIAL_CODES = {"facade": SP_FACADE, "revenge": SP_REVENGE, "flail": SP_FLAIL, "magnitude": SP_MAGNITUDE,
                 "smelling-salt": SP_SMELLING_SALT, "thaw-user": SP_THAW_USER, "rapid-spin": SP_RAPID_SPIN}
MAGNITUDE_BP = np.array([10, 30, 50, 70, 90, 110, 150], dtype=np.int32)
MAGNITUDE_CUM = np.array([5, 15, 35, 65, 85, 95, 100], dtype=np.int32)

# abilities (bit mask per Pokémon)
ABILITY_BITS = {name: 1 << i for i, name in enumerate((
    "limber", "immunity", "water-veil", "magma-armor", "insomnia", "vital-spirit", "own-tempo", "inner-focus",
    "shield-dust", "clear-body", "white-smoke", "hyper-cutter", "keen-eye", "synchronize", "shed-skin",
    "natural-cure", "static", "flame-body", "poison-point", "effect-spore", "guts", "intimidate", "rock-head",
    "compound-eyes", "damp", "early-bird", "sturdy", "serene-grace"))}
A = ABILITY_BITS
STAT_DROP_BLOCK = A["clear-body"] | A["white-smoke"]
CONTACT_ABILITIES = A["static"] | A["flame-body"] | A["poison-point"] | A["effect-spore"]
PINCH_TYPES = {"overgrow": "grass", "blaze": "fire", "torrent": "water", "swarm": "bug"}
IMMUNITY_ABILITIES = {"levitate": "ground", "flash-fire": "fire", "volt-absorb": "electric", "water-absorb": "water"}
# types (bit mask per Pokémon) that matter for status immunities
T_FIRE, T_ICE, T_POISON, T_STEEL = 1, 2, 4, 8
TYPE_BITS = {"fire": T_FIRE, "ice": T_ICE, "poison": T_POISON, "steel": T_STEEL}


def _ability_mask(name: str) -> int:
    return ABILITY_BITS.get(_norm(name), 0)


def _norm(name: str) -> str:
    return (name or "").lower().replace(" ", "-")


@dataclass(frozen=True)
class Rules:
    tie_to_rival: bool = False  # speed ties: the rival moves first (deterministic simulator) or 50 %
    recharge_on_miss: bool = False  # Hyper Beam needs recharging even if it did not hit (deterministic)
    # Quirk of the deterministic simulator: the recharge turn after Hyper Beam also counts as the
    # charging turn of a two-turn move, so Solar Beam fires right after recharging.
    charge_while_recharging: bool = False
    # secondary effects, status conditions, stat changes and abilities (off: deterministic simulator)
    mechanics: bool = True


RANDOM_RULES = Rules()
COMPAT_RULES = Rules(tie_to_rival=True, recharge_on_miss=True, charge_while_recharging=True, mechanics=False)


# --------------------------------------------------------------------------- scenarios


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class Scenarios:
    """``n`` random scenarios (from ``start``): for each one, trainer and turn, ``SLOTS`` uniform
    16-bit numbers. ``block`` has shape (trainers, n, T_CAP, SLOTS)."""

    def __init__(self, n: int, seed: int = 0, start: int = 0):
        self.n, self.seed, self.start = int(n), int(seed), int(start)
        t = np.arange(N_TRAINERS, dtype=np.uint64)[:, None, None, None]
        s = (np.arange(self.n, dtype=np.uint64) + np.uint64(self.start))[None, :, None, None]
        turn = np.arange(T_CAP, dtype=np.uint64)[None, None, :, None]
        word = np.arange(SLOTS // 4, dtype=np.uint64)[None, None, None, :]
        counter = (t << np.uint64(50)) | (s << np.uint64(14)) | (turn << np.uint64(4)) | word
        seed_mix = _splitmix64(np.array([self.seed], dtype=np.uint64))[0]
        words = _splitmix64(counter ^ seed_mix)
        self.block = words.view(np.uint16).reshape(N_TRAINERS, self.n, T_CAP, SLOTS)

    def __repr__(self):
        return f"Scenarios(n={self.n}, seed={self.seed}, start={self.start})"


# --------------------------------------------------------------------------- damage formula


def calc_stat(base: int, level: int) -> int:
    """Generation 3 stat with IV 31, no EVs and a neutral nature (as the damage server)."""
    return (2 * base + 31) * level // 100 + 5


def calc_hp(base: int, level: int) -> int:
    return (2 * base + 31) * level // 100 + level + 10


def low_kick_bp(weight: float) -> int:
    return 120 if weight >= 200 else 100 if weight >= 100 else 80 if weight >= 50 else 60 if weight >= 25 else \
        40 if weight >= 10 else 20


def gen3_damage(level: int, at: int, bp: int, df: int, physical: bool, stab: bool, eff4: int, crit: bool = False,
                burned: bool = False) -> int:
    """100 % roll of the generation 3 formula (as @smogon/calc) once the attack and defence stats
    already include their stages and abilities."""
    base = ((2 * level // 5 + 2) * at * bp // df) // 50
    if burned and physical:
        base //= 2
    base = (max(1, base) if physical else base) + 2
    if crit:
        base *= 2
    if stab:
        base = base * 3 // 2
    return max(1, base * eff4 // 4) if eff4 > 0 else 0  # at least 1 HP unless immune


_ROLL_FACTORS = tuple(range(85, 101))


@lru_cache(maxsize=None)
def _roll_stats(base: int) -> tuple:
    """(mean, lowest) of the 16 damage rolls of a move with this base damage."""
    if base <= 0:
        return 0.0, 0
    rolls = [max(1, base * r // 100) for r in _ROLL_FACTORS]
    return sum(rolls) / 16, rolls[0]


# --------------------------------------------------------------------------- moves


def _rival_abilities() -> dict:
    out = {}

    def walk(x):
        if isinstance(x, dict):
            if "species" in x:
                out.setdefault((x["species"], x["level"]), x.get("ability", ""))
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(TRAINERS)
    return out


class MoveStore:
    """Candidate moves of every matchup (damage key), as arrays indexed by ``[key_id, k]``, and the
    effects of every move (arrays indexed by move id).

    Candidates are ordered by preference for a sure KO (accuracy, priority, no recharge), keeping
    only those that add a new KO threshold; the move with the highest expected damage per turn is
    always included (``ev_best``). Each candidate keeps the inputs of the damage formula (attack
    and defence stat, power, effectiveness...) so that the damage can be recomputed with stat
    stages and status conditions."""

    FIELDS = ("move_id", "base", "crit_base", "acc", "crit_thr", "hits", "drain", "flags", "prio", "ko_min",
              "at", "df", "bp", "eff4")

    def __init__(self):
        self.key_ids: dict[str, int] = {}
        self.max_hp: list[int] = []
        self.level: list[int] = []
        self.rows: list[list] = []
        self.ev_best: list[int] = []
        self.move_ids: dict[str, int] = {}
        self.arrays = None
        self.validation = {"formula": 0, "table": 0}

    def _move_id(self, move: str) -> int:
        return self.move_ids.setdefault(move, len(self.move_ids))

    def _add(self, key: str, max_hp: int, level: int, rows: list, ev_best: int) -> None:
        self.key_ids[key] = len(self.rows)
        self.max_hp.append(max_hp)
        self.level.append(level)
        pad = (-1, 0, 0, NEVER_MISS, 0, 1, 0, 0, 0, -1, 1, 1, 0, 0)
        self.rows.append((rows + [pad] * K)[:K])
        self.ev_best.append(ev_best)

    def _finish(self, with_effects: bool) -> "MoveStore":
        data = np.array(self.rows, dtype=np.int32)  # (keys, K, fields)
        self.arrays = {name: np.ascontiguousarray(data[:, :, i]) for i, name in enumerate(self.FIELDS)}
        self.arrays["ev_best"] = np.array(self.ev_best, dtype=np.int32)
        self.arrays["max_hp"] = np.array(self.max_hp, dtype=np.int32)
        self.arrays["level"] = np.array(self.level, dtype=np.int32)
        # Move choice as a table [key, defender HP]: the first candidate (in KO preference order)
        # whose lowest damage KOs, otherwise the best expected damage per turn.
        hp = np.arange(int(self.arrays["max_hp"].max()) + 1)
        policy = np.empty((len(self.rows), len(hp)), dtype=np.int8)
        ko = self.arrays["ko_min"]
        for start in range(0, len(self.rows), 2048):
            block = ko[start:start + 2048, :, None] >= hp[None, None, :]
            first = block.argmax(1)
            has = np.take_along_axis(block, first[:, None, :], 1)[:, 0, :]
            policy[start:start + 2048] = np.where(has, first, self.arrays["ev_best"][start:start + 2048, None])
        self.arrays["policy"] = policy
        self._build_effects(with_effects)
        return self

    def _build_effects(self, with_effects: bool) -> None:
        n = max(1, len(self.move_ids))
        eff = {name: np.zeros(n, np.int32) for name in (
            "ail", "ail_ch", "flinch", "tstat", "tdelta", "tch", "sch", "special", "fire", "semi_kind", "hits_semi",
            "double_semi")}
        eff["tstat"][:] = -1
        self_delta = np.zeros((n, 7), np.int32)
        names = [""] * n
        for move, mid in self.move_ids.items():
            names[mid] = move
            md = MOVE_DATA.get(move)
            if md is None or not with_effects:
                continue
            ail = md.get("ailment", "none")
            eff["ail"][mid] = {"confusion": AIL_CONFUSION, "tri": AIL_TRI, "trap": AIL_TRAP}.get(
                ail, STATUS_CODES.get(ail, NONE))
            eff["ail_ch"][mid] = md.get("ailment_chance", 0)
            eff["flinch"][mid] = md.get("flinch", 0)
            if md.get("stat_target"):
                stat, delta = md["stat_target"][0]
                eff["tstat"][mid], eff["tdelta"][mid], eff["tch"][mid] = STAT_INDEX[stat], delta, md["stat_chance"]
            for stat, delta in md.get("stat_self", []):
                self_delta[mid, STAT_INDEX[stat]] = delta
            eff["sch"][mid] = md.get("self_chance", 0)
            eff["special"][mid] = SPECIAL_CODES.get(md.get("special", ""), 0)
            eff["fire"][mid] = md.get("type") == "fire"
            eff["semi_kind"][mid] = SEMI_KIND.get(move, 0)
            eff["hits_semi"][mid] = HITS_SEMI.get(move, 0)
            eff["double_semi"][mid] = move in DOUBLE_SEMI
        self.effects = eff
        self.effects["self_delta"] = self_delta
        eff["any"] = ((eff["ail"] > 0) | (eff["flinch"] > 0) | (eff["tstat"] >= 0) | (eff["sch"] > 0)
                      | (eff["special"] == SP_RAPID_SPIN))
        self.move_names = names

    def __getitem__(self, name):
        return self.arrays[name]

    def key_id(self, key: str) -> int:
        try:
            return self.key_ids[key]
        except KeyError:
            raise KeyError(f"Falta el enfrentamiento {key} en la tabla de tiradas: "
                           "ejecuta 'python tools/precompute_damage.py --rolls'") from None

    @classmethod
    def from_rolls(cls, rolls: dict, priorities: dict) -> "MoveStore":
        species = rolls.get("__species__", {})
        chart = rolls.get("__types__", {})
        rival_ab = _rival_abilities()
        store = cls()
        for key, entry in rolls.items():
            if key.startswith("__"):
                continue
            max_hp, moves = entry
            parts = key.split("|")
            att, att_lvl, dfn, def_lvl = parts[0], int(parts[1]), parts[2], int(parts[3])
            rival_attacking = len(parts) > 4
            sa, sd = species.get(att), species.get(dfn)
            if rival_attacking:
                att_ab, def_ab = rival_ab.get((att, att_lvl), ""), (sd or {}).get("ability", "")
            else:
                att_ab, def_ab = (sa or {}).get("ability", ""), rival_ab.get((dfn, def_lvl), "")
            att_ab, def_ab = _norm(att_ab), _norm(def_ab)
            no_crit = def_ab in ("battle-armor", "shell-armor")
            cands = []
            for move, base, crit_base, fixed in moves:
                if move in UNUSABLE_MOVES:
                    continue
                md = MOVE_DATA.get(move, {"accuracy": 100, "crit_stage": 0, "hits": [1, 1], "drain": 0})
                acc = md["accuracy"] if md["accuracy"] is not None else NEVER_MISS
                thr = 0 if no_crit else CRIT_THRESHOLD[min(1, md["crit_stage"])]
                hits_min, hits_max = md["hits"]
                hits = 1 if hits_max == 1 else (2 if hits_max == 2 else 5)
                mean_hits = 1.0 if hits == 1 else (2.0 if hits == 2 else MEAN_HITS_2_5)
                phys = md.get("category") == "physical"
                mtype = md.get("type", "normal")
                flags = (F_CHARGE * (move in CHARGE_MOVES) | F_SEMI * (move in SEMI_INVULNERABLE_MOVES)
                         | F_RECHARGE * (move in RECHARGE_MOVES) | F_CRASH * md.get("crash", False)
                         | F_SELF_KO * md.get("self_ko", False) | F_FIXED * bool(fixed)
                         | F_HALF_HP * (move == "super-fang") | F_ENDEAVOR * (move == "endeavor")
                         | F_NO_KO * (move == "false-swipe") | F_OHKO * (md.get("special") == "ohko")
                         | F_PHYS * phys | F_CONTACT * md.get("contact", False) | F_LOCK * md.get("lock", False))
                # inputs of the damage formula
                at = df = 1
                bp = eff4 = 0
                if sa and sd and chart:
                    at = calc_stat(sa["baseStats"]["atk" if phys else "spa"], att_lvl)
                    df = calc_stat(sd["baseStats"]["def" if phys else "spd"], def_lvl)
                    if def_ab == "thick-fat" and mtype in ("fire", "ice"):
                        at //= 2
                    if move in ("self-destruct", "explosion"):
                        df //= 2
                    bp = md.get("bp", 0)
                    if md.get("special") == "low-kick":
                        bp = low_kick_bp(sd["weightkg"])
                    elif md.get("special") == "flail":
                        bp = 20  # full HP; recomputed during the battle
                    eff = 1.0
                    for t in sd["types"]:
                        eff *= chart.get(mtype.capitalize(), {}).get(t.capitalize(), 1)
                    if IMMUNITY_ABILITIES.get(def_ab) == mtype:
                        eff = 0
                    eff4 = int(round(eff * 4))
                    if mtype in sa["types"]:
                        flags |= F_STAB
                    if PINCH_TYPES.get(att_ab) == mtype:
                        flags |= F_PINCH
                    special_mode = flags & (F_FIXED | F_HALF_HP | F_ENDEAVOR | F_OHKO) or md.get("special") == "magnitude"
                    if not special_mode:
                        normal = gen3_damage(att_lvl, at, bp, df, phys, bool(flags & F_STAB), eff4)
                        crit = gen3_damage(att_lvl, at, bp, df, phys, bool(flags & F_STAB), eff4, crit=True)
                        if normal == base and (no_crit or crit == crit_base):
                            store.validation["formula"] += 1
                        else:
                            flags |= F_TABLE  # use the damage of the calculator (stat stages approximated)
                            store.validation["table"] += 1
                else:
                    flags |= F_TABLE
                if flags & F_FIXED:
                    mean, low = float(base), base
                elif flags & F_OHKO:
                    mean, low = max_hp * max(0, 30 + att_lvl - def_lvl) / 100, 0
                    acc = NEVER_MISS  # the accuracy check of one-hit KO moves is special
                elif md.get("special") == "magnitude":
                    mean, low = float(base), 0
                else:
                    pc = thr / 65536
                    mean = _roll_stats(base)[0] * (1 - pc) + _roll_stats(crit_base)[0] * pc
                    low = _roll_stats(base)[1]
                if move in ("super-fang", "endeavor"):
                    mean, low = max_hp / 4, 0  # depends on the current HP: never a "sure" KO
                rate = (min(acc, 100) / 100) * mean * mean_hits / (2 if flags & (F_CHARGE | F_RECHARGE) else 1)
                if flags & F_OHKO:
                    rate = mean
                ko_ok = low > 0 and not flags & (F_CHARGE | F_NO_KO)
                ko_min = low * hits_min if ko_ok else -1
                prio = priorities.get(move, 0)
                cands.append({"row": (store._move_id(move), base, crit_base, acc, thr, hits, md["drain"], flags, prio,
                                      ko_min, at, df, bp, eff4),
                              "rate": rate, "acc": acc, "prio": prio, "ko_min": ko_min, "flags": flags})
            if not cands:
                cands = [{"row": (store._move_id("struggle"), 0, 0, NEVER_MISS, 0, 1, 0, F_TABLE, 0, -1, 1, 1, 0, 0),
                          "rate": 0.0, "acc": NEVER_MISS, "prio": 0, "ko_min": -1, "flags": F_TABLE}]
            best = max(cands, key=lambda c: c["rate"])
            order = sorted((c for c in cands if c["ko_min"] > 0),
                           key=lambda c: (-c["acc"], -c["prio"], bool(c["flags"] & (F_RECHARGE | F_SELF_KO | F_CRASH)),
                                          -c["rate"]))
            kept, threshold = [], 0
            for c in order:
                if c["ko_min"] > threshold:  # KOs in cases where no preferred move does
                    kept.append(c)
                    threshold = c["ko_min"]
            kept = kept[:K - 1]
            if best not in kept:
                kept.append(best)
            store._add(key, max_hp, att_lvl, [c["row"] for c in kept], kept.index(best))
        store.species = species
        return store._finish(with_effects=True)

    @classmethod
    def from_deterministic(cls, table: dict, priorities: dict) -> "MoveStore":
        """One move per matchup with the damage of the deterministic table (% of HP, max HP = 100):
        replays battle_simulator.simulate_battle exactly with ``COMPAT_RULES``."""
        store = cls()
        for key, value in table.items():
            if key == bs.FORMAT_KEY:
                continue
            move, lost = value
            dmg = round(lost * 100)
            flags = (F_CHARGE * (move in bs.CHARGE_MOVES) | F_SEMI * (move in bs.SEMI_INVULNERABLE_MOVES)
                     | F_RECHARGE * (move == "hyper-beam") | F_FIXED)
            drain = 50 if move in bs.DRAIN_MOVES else 0
            ko_min = dmg if dmg > 0 and not flags & F_CHARGE else -1
            store._add(key, 100, int(key.split("|")[1]),
                       [(store._move_id(move), dmg, dmg, NEVER_MISS, 0, 1, drain, flags, priorities.get(move, 0),
                         ko_min, 1, 1, 0, 0)], 0)
        return store._finish(with_effects=False)


# --------------------------------------------------------------------------- battles


@dataclass
class BattleSpec:
    trainer: int
    n: int
    m: int
    keys: np.ndarray  # (6, 6, 2) key ids: [player i, rival j, side]
    p_max: np.ndarray  # (6,) max HP of the player's team
    r_max: np.ndarray  # (6,)
    p_speed: np.ndarray  # (6,)
    r_speed: np.ndarray  # (6,)
    plan: np.ndarray  # (6,) player position that must face each rival Pokémon, -1 = keep
    team: list  # evolved team (pkm ids)
    level_cap: int = 0
    rival_team: list = field(default_factory=list)
    p_level: np.ndarray = None  # (6,)
    r_level: np.ndarray = None
    p_ability: np.ndarray = None  # (6,) ability bit masks
    r_ability: np.ndarray = None
    p_types: np.ndarray = None  # (6,) type bit masks
    r_types: np.ndarray = None
    p_atk: np.ndarray = None  # (6,) attack and defence (confusion damage)
    p_def: np.ndarray = None
    r_atk: np.ndarray = None
    r_def: np.ndarray = None


def _species_data(store: MoveStore) -> dict:
    return getattr(store, "species", {}) or {}


def compile_battle(genome, trainer_index: int, store: MoveStore, dataset: dict | None = None) -> BattleSpec:
    dataset = dataset if dataset is not None else bs.dataset
    pkdex = dataset["pkdex"]
    catches, teams = genome[0], genome[1]
    plans = genome[2] if len(genome) > 2 else None
    plan = plans[trainer_index] if plans else None
    name = TRAINERS_ORDER[trainer_index]
    rival = bs.get_rival_team(trainer_index, catches[0])
    cap = bs.select_level_cap(rival)
    base_team = list(teams[trainer_index])
    team = [bs.evolve_pokemon(p, cap, AVAILABLE_EVOLVE_OBJ_TRAINERS[name], dataset) for p in base_team]
    n, m = len(team), len(rival)
    z = lambda: np.zeros(MAX_TEAM, dtype=np.int32)  # noqa: E731
    keys = np.zeros((MAX_TEAM, MAX_TEAM, 2), dtype=np.int32)
    p_max, r_max, p_speed, r_speed = z(), z(), z(), z()
    p_level, r_level, p_ab, r_ab, p_ty, r_ty = z(), z(), z(), z(), z(), z()
    p_atk, p_def, r_atk, r_def = z() + 1, z() + 1, z() + 1, z() + 1
    max_hp = store["max_hp"]
    species = _species_data(store)
    for i, pid in enumerate(team):
        entry = pkdex[str(pid)]
        p_speed[i] = bs.calculate_speed(entry["speed"], cap)
        p_level[i] = cap
        sp = species.get(entry["name"])
        if sp:
            p_ab[i] = _ability_mask(sp.get("ability", ""))
            p_ty[i] = sum(TYPE_BITS.get(t, 0) for t in sp["types"])
            p_atk[i] = calc_stat(sp["baseStats"]["atk"], cap)
            p_def[i] = calc_stat(sp["baseStats"]["def"], cap)
        for j, r in enumerate(rival):
            kp = store.key_id(bs.damage_key(entry["name"], cap, r["species"], r["level"], False))
            kr = store.key_id(bs.damage_key(r["species"], r["level"], entry["name"], cap, True))
            keys[i, j, 0], keys[i, j, 1] = kp, kr
            r_max[j], p_max[i] = max_hp[kp], max_hp[kr]
    for j, r in enumerate(rival):
        r_speed[j] = r["speed"]
        r_level[j] = r["level"]
        r_ab[j] = _ability_mask(r.get("ability", ""))
        sp = species.get(r["species"])
        if sp:
            r_ty[j] = sum(TYPE_BITS.get(t, 0) for t in sp["types"])
            r_atk[j] = calc_stat(sp["baseStats"]["atk"], r["level"])
            r_def[j] = calc_stat(sp["baseStats"]["def"], r["level"])
    planned = np.full(MAX_TEAM, -1, dtype=np.int32)
    for j, pos in enumerate(bs.resolve_plan(plan, base_team, m)):
        if pos is not None:
            planned[j] = pos
    return BattleSpec(trainer_index, n, m, keys, p_max, r_max, p_speed, r_speed, planned, team, cap, rival,
                      p_level, r_level, p_ab, r_ab, p_ty, r_ty, p_atk, p_def, r_atk, r_def)


@dataclass
class BatchResult:
    """Per battle and scenario: shape (battles, scenarios)."""
    won: np.ndarray
    hp_lost: np.ndarray  # % of HP lost by the player's team (sum over its Pokémon)
    rival_hp_left: np.ndarray  # % of HP the rival team still has
    turns: np.ndarray
    switches: np.ndarray
    timed_out: np.ndarray
    events: list | None = None  # [battle][scenario] -> list of events (record=True)


def simulate(specs: list, scenarios: Scenarios, store: MoveStore, rules: Rules = RANDOM_RULES,
             max_lanes: int = 100_000, record: bool = False) -> BatchResult:
    """Plays every battle of ``specs`` once per scenario. ``record`` keeps the events of every
    battle (only for small batches: the viewer)."""
    B, N = len(specs), scenarios.n
    out = BatchResult(np.zeros((B, N), bool), np.zeros((B, N), np.float32), np.zeros((B, N), np.float32),
                      np.zeros((B, N), np.int16), np.zeros((B, N), np.int16), np.zeros((B, N), bool))
    if record:
        out.events = [[[] for _ in range(N)] for _ in range(B)]
    if B == 0:
        return out
    per_chunk = max(1, max_lanes // N)
    for b0 in range(0, B, per_chunk):
        _simulate_chunk(specs[b0:b0 + per_chunk], scenarios, store, rules, out, b0)
    return out


def _stage(value, stage):
    return value * STAGE_NUM[stage + 6] // STAGE_DEN[stage + 6]


def _pct(hp: int, mx: int) -> int:
    return 0 if hp <= 0 else max(1, int(round(hp * 100 / max(1, mx))))


def _simulate_chunk(specs, scenarios, store, rules, out, b0):  # noqa: C901 (one turn of the battle)
    N = scenarios.n
    mech = rules.mechanics
    recording = out.events is not None
    S_keys = np.stack([s.keys for s in specs]).reshape(-1)  # [((b * 6 + i) * 6 + j) * 2 + side]
    st = lambda name: np.stack([getattr(s, name) if getattr(s, name) is not None else np.zeros(MAX_TEAM, np.int32)  # noqa: E731
                                for s in specs])
    S_pmax, S_rmax, S_plan = st("p_max"), st("r_max"), st("plan")
    F_pspd, F_rspd = st("p_speed").reshape(-1), st("r_speed").reshape(-1)
    F_plvl, F_rlvl = st("p_level").reshape(-1), st("r_level").reshape(-1)
    F_pab, F_rab = st("p_ability").reshape(-1), st("r_ability").reshape(-1)
    F_pty, F_rty = st("p_types").reshape(-1), st("r_types").reshape(-1)
    F_patk, F_pdef = st("p_atk").reshape(-1), st("p_def").reshape(-1)
    F_ratk, F_rdef = st("r_atk").reshape(-1), st("r_def").reshape(-1)
    if not mech:
        F_pab, F_rab = np.zeros_like(F_pab), np.zeros_like(F_rab)
    S_m = np.array([s.m for s in specs], dtype=np.int32)
    S_n = np.array([s.n for s in specs], dtype=np.int32)
    S_t = np.array([s.trainer for s in specs], dtype=np.int64)

    # flat views: 1-D indexing ([key * K + move]) is about twice as fast as 2-D fancy indexing
    fa = lambda f: store[f].reshape(-1)  # noqa: E731
    A_base, A_crit, A_acc, A_thr = fa("base"), fa("crit_base"), fa("acc"), fa("crit_thr")
    A_hits, A_drain, A_flags, A_prio = fa("hits"), fa("drain"), fa("flags"), fa("prio")
    A_at, A_df, A_bp, A_eff4, A_mid = fa("at"), fa("df"), fa("bp"), fa("eff4"), fa("move_id")
    A_level, POLICY, A_midk = store["level"], store["policy"], store["move_id"]
    E = store.effects
    RT = scenarios.block.reshape(-1, SLOTS)

    # lane state
    L = len(specs) * N
    lb = (np.arange(L) // N).astype(np.int32)  # battle (in this chunk)
    ls = (np.arange(L) % N).astype(np.int32)  # scenario
    rrow = ((S_t[lb] * N + ls) * T_CAP).astype(np.int32)  # row of the lane's random numbers for turn 0
    p_hp = S_pmax[lb].copy()
    r_hp = S_rmax[lb].copy()
    F_pmax, F_rmax = S_pmax.reshape(-1), S_rmax.reshape(-1)
    p_alive = S_n[lb].copy()
    p_idx = np.zeros(L, np.int32)
    r_idx = np.zeros(L, np.int32)
    p_status = np.zeros((L, MAX_TEAM), np.int8)
    r_status = np.zeros((L, MAX_TEAM), np.int8)
    p_sleep = np.zeros((L, MAX_TEAM), np.int8)
    r_sleep = np.zeros((L, MAX_TEAM), np.int8)
    charging = np.full((L, 2), -1, np.int16)  # move id being charged
    semi = np.zeros((L, 2), bool)
    recharge = np.zeros((L, 2), bool)
    stages = np.zeros((L, 2, 7), np.int8)
    confusion = np.zeros((L, 2), np.int8)
    trap = np.zeros((L, 2), np.int8)  # end-of-turn counter while trapped (Bind, Wrap...)
    lock = np.zeros((L, 2), np.int8)  # turns left of a rampage (Thrash)
    lock_mid = np.full((L, 2), -1, np.int16)
    tox_n = np.zeros((L, 2), np.int8)
    switches = np.zeros(L, np.int32)
    tgt = S_plan[lb, 0]
    pending = np.where(tgt > 0, tgt, -1)  # planned member for the first rival (position 0 already leads)
    lane_id = lb * N + ls  # for the recorder

    # ------------------------------------------------------------------ recorder
    names_p = [[bs.dataset["pkdex"][str(p)]["name"] for p in s.team] for s in specs]
    names_r = [[r["species"] for r in s.rival_team] for s in specs]

    cur = {"hpa": None, "sta": None}  # active HP / status during a turn (not yet written back)

    def snap(i, hpa_=None, sta_=None):
        hpa_ = cur["hpa"] if hpa_ is None else hpa_
        sta_ = cur["sta"] if sta_ is None else sta_
        ph, rh = p_hp[i].copy(), r_hp[i].copy()
        ps, rs = p_status[i].copy(), r_status[i].copy()
        if hpa_ is not None:
            ph[p_idx[i]], rh[r_idx[i]] = hpa_[i, 0], hpa_[i, 1]
        if sta_ is not None:
            ps[p_idx[i]], rs[r_idx[i]] = sta_[i, 0], sta_[i, 1]
        b = lb[i]
        n_, m_ = specs[b].n, specs[b].m
        r = min(r_idx[i], m_ - 1)
        p_hps = [_pct(ph[k], S_pmax[b, k]) for k in range(n_)]
        r_hps = [_pct(rh[k], S_rmax[b, k]) for k in range(m_)]
        return {"p_idx": int(p_idx[i]), "p_id": specs[b].team[p_idx[i]], "p_hp": p_hps[p_idx[i]], "r_idx": int(r),
                "r_species": names_r[b][r], "r_hp": r_hps[r], "p_hps": p_hps, "r_hps": r_hps,
                "dt": n_ * 100 - sum(p_hps), "dr": m_ * 100 - sum(r_hps),
                "p_status": STATUS_NAMES[ps[p_idx[i]]], "r_status": STATUS_NAMES[rs[r]],
                "p_statuses": [STATUS_NAMES[x] for x in ps[:n_]], "r_statuses": [STATUS_NAMES[x] for x in rs[:m_]]}

    def who(i, side):
        b = lb[i]
        return names_p[b][p_idx[i]] if side == 0 else names_r[b][min(r_idx[i], specs[b].m - 1)]

    def rec(lanes, etype, hpa_=None, sta_=None, **cols):
        if not recording:
            return
        for n_, i in enumerate(np.atleast_1d(lanes)):
            ev = {"type": etype}
            for k, v in cols.items():
                val = v[n_] if isinstance(v, (np.ndarray, list)) else v
                ev[k] = val.item() if isinstance(val, np.generic) else val
            if "side" in ev and isinstance(ev["side"], int):
                ev["pokemon"] = who(i, ev["side"])
                ev["side"] = bs.SIDE_NAMES[ev["side"]]
            ev.update(snap(i, hpa_, sta_))
            b = lb[i]
            out.events[b0 + b][ls[i]].append(ev)

    # ------------------------------------------------------------------ helpers (mechanics)

    def abil(lanes):
        """Abilities of the active Pokémon of ``lanes`` (L, 2)."""
        ri = np.minimum(r_idx[lanes], MAX_TEAM - 1)
        return np.stack([F_pab[lb[lanes] * MAX_TEAM + p_idx[lanes]], F_rab[lb[lanes] * MAX_TEAM + ri]], 1)

    def intimidate(lanes, side, ab_side, ab_target):
        """Intimidate of the Pokémon entering on ``side`` lowers the attack of the opponent."""
        if not mech or not len(lanes):
            return
        has = (ab_side & A["intimidate"]) != 0
        lanes = lanes[has]
        if not len(lanes):
            return
        o = 1 - side
        blocked = (ab_target[has] & (STAT_DROP_BLOCK | A["hyper-cutter"])) != 0
        ok = ~blocked & (stages[lanes, o, ATK] > -6)
        stages[lanes[ok], o, ATK] -= 1
        if recording:
            rec(lanes, "ability", side=side, ability="intimidate", target_side=bs.SIDE_NAMES[o],
            target=[who(i, o) for i in lanes], effective=ok)

    def entry_abilities(lanes_p, lanes_r):
        """Abilities that trigger when a Pokémon enters (lanes where the player / rival Pokémon is new)."""
        if not mech:
            return
        if len(lanes_p):
            ab_ = abil(lanes_p)
            intimidate(lanes_p, 0, ab_[:, 0], ab_[:, 1])
        if len(lanes_r):
            ab_ = abil(lanes_r)
            intimidate(lanes_r, 1, ab_[:, 1], ab_[:, 0])

    def reset_side(lanes, side):
        charging[lanes, side] = -1
        semi[lanes, side] = False
        recharge[lanes, side] = False
        stages[lanes, side, :] = 0
        confusion[lanes, side] = 0
        lock[lanes, side] = 0
        lock_mid[lanes, side] = -1
        tox_n[lanes, side] = 0
        trap[lanes, side] = 0
        trap[lanes, 1 - side] = 0  # the trapper leaves: the opponent is freed

    rnd_box = [None]

    def rnd(lanes, side, slot):
        return rnd_box[0][lanes * SLOTS + side * SIDE_SLOTS + slot].astype(np.int32)

    def chance(lanes, side, slot, pct):
        return ((rnd(lanes, side, slot) * 100) >> 16) < pct

    def try_status(lanes, side, code, source):
        """Status ``code`` on the active Pokémon of ``side`` (arrays per lane), inflicted by
        ``source``; type and ability immunities. Synchronize passes it back. Returns the mask."""
        hpa, sta = cur["hpa"], cur["sta"]
        t_ab, t_ty = ab_box[0][lanes, side], ty_box[0][lanes, side]
        ok = (sta[lanes, side] == NONE) & (hpa[lanes, side] > 0) & (code != NONE)
        ok &= ~((code == BRN) & (((t_ty & T_FIRE) != 0) | ((t_ab & A["water-veil"]) != 0)))
        ok &= ~((code == FRZ) & (((t_ty & T_ICE) != 0) | ((t_ab & A["magma-armor"]) != 0)))
        ok &= ~(((code == PSN) | (code == TOX)) & (((t_ty & (T_POISON | T_STEEL)) != 0) | ((t_ab & A["immunity"]) != 0)))
        ok &= ~((code == PAR) & ((t_ab & A["limber"]) != 0))
        ok &= ~((code == SLP) & ((t_ab & (A["insomnia"] | A["vital-spirit"])) != 0))
        j = np.nonzero(ok)[0]
        if not len(j):
            return ok
        li, sd, c, src = lanes[j], side[j], code[j], source[j]
        sta[li, sd] = c
        sl = c == SLP
        if sl.any():
            slp_box[0][li[sl], sd[sl]] = 2 + ((rnd(li[sl], src[sl], S_DUR) >> 9) & 3)
        tx = c == TOX
        tox_n[li[tx], sd[tx]] = 0
        if recording:
            rec(li, "status", side=sd, status=[STATUS_NAMES[x] for x in c])
        sync = ((t_ab[j] & A["synchronize"]) != 0) & np.isin(c, (PAR, BRN, PSN, TOX))
        if sync.any():
            k = np.nonzero(sync)[0]
            back = np.where(c[k] == TOX, PSN, c[k])
            hpa_s, sta_s = cur["hpa"], cur["sta"]
            o_ab, o_ty = ab_box[0][li[k], src[k]], ty_box[0][li[k], src[k]]
            ok2 = (sta_s[li[k], src[k]] == NONE) & (hpa_s[li[k], src[k]] > 0)
            ok2 &= ~((back == BRN) & (((o_ty & T_FIRE) != 0) | ((o_ab & A["water-veil"]) != 0)))
            ok2 &= ~((back == PSN) & (((o_ty & (T_POISON | T_STEEL)) != 0) | ((o_ab & A["immunity"]) != 0)))
            ok2 &= ~((back == PAR) & ((o_ab & A["limber"]) != 0))
            if ok2.any():
                sta_s[li[k][ok2], src[k][ok2]] = back[ok2]
                if recording:
                    rec(li[k][ok2], "ability", side=sd[k][ok2], ability="synchronize")
                    rec(li[k][ok2], "status", side=src[k][ok2], status=[STATUS_NAMES[x] for x in back[ok2]])
        return ok

    def change_stats(lanes, side, stat, delta, by_opponent):
        """Stage changes (arrays per lane). Drops by the opponent are blocked by Clear Body, White
        Smoke, Hyper Cutter (attack) and Keen Eye (accuracy)."""
        t_ab = ab_box[0][lanes, side]
        blocked = np.zeros(len(lanes), bool)
        if by_opponent:
            drop = delta < 0
            blocked = drop & (((t_ab & STAT_DROP_BLOCK) != 0) | ((stat == ATK) & ((t_ab & A["hyper-cutter"]) != 0))
                              | ((stat == ACC) & ((t_ab & A["keen-eye"]) != 0)))
        old = stages[lanes, side, stat]
        new = np.clip(old + delta, -6, 6)
        ok = ~blocked & (new != old)
        stages[lanes[ok], side[ok], stat[ok]] = new[ok]
        if recording:
            rec(lanes, "stat", side=side, stat=[STAT_NAMES[x] for x in stat], delta=delta, applied=ok,
                blocked=blocked)

    def effects_after_hit(i, aa, dd, mv, fl, hit, real, h_att, h_def, ab_a, ab_d, phase):
        """Secondary effects of the move, effects on the user and contact abilities."""
        sta = cur["sta"]
        landed = hit & (real > 0)
        has_effect = E["any"][mv] | (((fl & F_CONTACT) != 0) & ((ab_d & CONTACT_ABILITIES) != 0))
        sta_t = sta[i, dd]
        has_effect |= ((sta_t == FRZ) & (E["fire"][mv] == 1)) | ((sta_t == PAR) & (E["special"][mv] == SP_SMELLING_SALT))
        sel = np.nonzero(has_effect & hit)[0]
        if not len(sel):
            return
        i, aa, dd, mv, fl, hit, real = i[sel], aa[sel], dd[sel], mv[sel], fl[sel], hit[sel], real[sel]
        h_att, h_def, ab_a, ab_d, landed = h_att[sel], h_def[sel], ab_a[sel], ab_d[sel], landed[sel]
        can_t = landed & (h_def > 0) & ((ab_d & A["shield-dust"]) == 0)
        serene = (ab_a & A["serene-grace"]) != 0
        sec_u = (rnd(i, aa, S_SEC) * 100) >> 16

        def roll(ch):
            return sec_u < np.where(serene, ch * 2, ch)

        ail = E["ail"][mv]
        j = np.nonzero(can_t & (ail > 0) & roll(E["ail_ch"][mv]))[0]
        if len(j):
            li, sd, sa_, aj = i[j], dd[j], aa[j], ail[j]
            tri = np.array([PAR, BRN, FRZ], np.int32)[(rnd(li, sa_, S_SEC_KIND) * 3) >> 16]
            code = np.where(aj == AIL_TRI, tri, np.where(aj <= SLP, aj, NONE))
            k = np.nonzero(code != NONE)[0]
            if len(k):
                try_status(li[k], sd[k], code[k], sa_[k])
            k = np.nonzero(aj == AIL_CONFUSION)[0]
            if len(k):
                ok = (confusion[li[k], sd[k]] == 0) & ((ab_box[0][li[k], sd[k]] & A["own-tempo"]) == 0)
                kk_ = k[ok]
                confusion[li[kk_], sd[kk_]] = 2 + ((rnd(li[kk_], sa_[kk_], S_DUR) >> 13) & 3)
                if recording:
                    rec(li[kk_], "confuse", side=sd[kk_], cause="move")
            k = np.nonzero(aj == AIL_TRAP)[0]
            if len(k):
                ok = trap[li[k], sd[k]] == 0
                kk_ = k[ok]
                trap[li[kk_], sd[kk_]] = 3 + ((rnd(li[kk_], sa_[kk_], S_DUR) >> 11) & 3)
                if recording:
                    rec(li[kk_], "trap", side=sd[kk_], move=[store.move_names[x] for x in mv[j][kk_]])
        fch = E["flinch"][mv]
        j = np.nonzero(can_t & (fch > 0) & roll(fch) & ((ab_d & A["inner-focus"]) == 0))[0]
        if len(j) and phase == 0:
            flinch_box[0][i[j], dd[j]] = True
        ts = E["tstat"][mv]
        j = np.nonzero(can_t & (ts >= 0) & roll(E["tch"][mv]))[0]
        if len(j):
            change_stats(i[j], dd[j], ts[j], E["tdelta"][mv[j]], True)
        sch = E["sch"][mv]
        j = np.nonzero(hit & (sch > 0) & (h_att > 0) & roll(sch))[0]
        if len(j):
            deltas = E["self_delta"][mv[j]]
            for stat in range(7):
                k = np.nonzero(deltas[:, stat] != 0)[0]
                if len(k):
                    change_stats(i[j[k]], aa[j[k]], np.full(len(k), stat, np.int32), deltas[k, stat], False)
        j = np.nonzero(hit & (E["special"][mv] == SP_RAPID_SPIN))[0]
        trap[i[j], aa[j]] = 0
        j = np.nonzero(landed & (h_def > 0) & (E["fire"][mv] == 1) & (sta[i, dd] == FRZ))[0]
        if len(j):  # Fire moves thaw the target
            sta[i[j], dd[j]] = NONE
            if recording:
                rec(i[j], "thaw", side=dd[j])
        j = np.nonzero(landed & (h_def > 0) & (E["special"][mv] == SP_SMELLING_SALT) & (sta[i, dd] == PAR))[0]
        if len(j):
            sta[i[j], dd[j]] = NONE
            if recording:
                rec(i[j], "cure", side=dd[j], cause="smelling-salt")
        contact = landed & ((fl & F_CONTACT) != 0) & (h_att > 0)
        if contact.any():
            third = ((rnd(i, aa, S_CONTACT) * 3) >> 16) == 0
            for name, code in (("static", PAR), ("flame-body", BRN), ("poison-point", PSN)):
                j = np.nonzero(contact & third & ((ab_d & A[name]) != 0))[0]
                if len(j):
                    if recording:
                        rec(i[j], "ability", side=dd[j], ability=name)
                    try_status(i[j], aa[j], np.full(len(j), code, np.int32), dd[j])
            spore = contact & ((ab_d & A["effect-spore"]) != 0) & (((rnd(i, aa, S_CONTACT) * 100) >> 16) < 10)
            j = np.nonzero(spore)[0]
            if len(j):
                code = np.array([SLP, PSN, PAR], np.int32)[(rnd(i[j], aa[j], S_CONTACT_KIND) * 3) >> 16]
                if recording:
                    rec(i[j], "ability", side=dd[j], ability="effect-spore")
                try_status(i[j], aa[j], code, dd[j])

    def end_of_turn(ar):
        """Shed Skin, poison, bad poison, burn and trapping damage (player first)."""
        hpa, sta, maxa = cur["hpa"], cur["sta"], cur["maxa"]
        if not (sta.any() or trap.any()):
            return
        for side in (0, 1):
            sides = np.full(len(ar), side, np.int32)
            alive = hpa[:, side] > 0
            shed = alive & (sta[:, side] != NONE) & ((ab_box[0][:, side] & A["shed-skin"]) != 0) \
                & (((rnd(ar, sides, S_SHED) * 3) >> 16) == 0)
            if shed.any():
                j = np.nonzero(shed)[0]
                sta[j, side] = NONE
                slp_box[0][j, side] = 0
                if recording:
                    rec(j, "cure", side=sides[j], cause="shed-skin")
            mx = maxa[:, side]
            st_ = sta[:, side]
            tox = alive & (st_ == TOX)
            tox_n[tox, side] = np.minimum(15, tox_n[tox, side] + 1)
            dmg = np.where(alive & ((st_ == PSN) | (st_ == BRN)), np.maximum(1, mx // 8), 0)
            dmg = np.where(tox, np.maximum(1, mx * tox_n[:, side] // 16), dmg)
            if dmg.any():
                hpa[:, side] = np.maximum(0, hpa[:, side] - dmg)
                if recording:
                    j = np.nonzero(dmg > 0)[0]
                    rec(j, "residual", side=sides[j], cause=[STATUS_NAMES[x] for x in st_[j]],
                        amount=[_pct(d, x) for d, x in zip(dmg[j], mx[j])], fainted=hpa[j, side] == 0)
            tr = (hpa[:, side] > 0) & (trap[:, side] > 0)
            if tr.any():
                trap[tr, side] -= 1
                hurt_ = tr & (trap[:, side] > 0)
                dmg = np.where(hurt_, np.maximum(1, mx // 16), 0)
                hpa[:, side] = np.maximum(0, hpa[:, side] - dmg)
                if recording:
                    j = np.nonzero(hurt_)[0]
                    rec(j, "residual", side=sides[j], cause="trap", amount=[_pct(d, x) for d, x in zip(dmg[j], mx[j])],
                        fainted=hpa[j, side] == 0)
                    j = np.nonzero(tr & (trap[:, side] == 0))[0]
                    rec(j, "trap_end", side=sides[j])

    ab_box, ty_box, slp_box, flinch_box = [None], [None], [None], [None]
    entry_abilities(np.arange(L), np.arange(L))

    turn = 0
    while L:
        turn += 1
        ar = np.arange(L)
        if turn > bs.MAX_TURNS:
            _store_results(out, b0, lb, ls, p_hp, S_pmax[lb], r_hp, S_rmax[lb], switches, bs.MAX_TURNS,
                           np.zeros(L, bool), np.ones(L, bool))
            for i in ar:
                rec(i, "end", won=False, timed_out=True, hp_lost=0)
            break
        R = RT[rrow + (turn - 1) % T_CAP]  # (L, SLOTS) uint16
        rnd_box[0] = R.reshape(-1)
        if recording:
            rec(ar, "turn", turn=turn)

        # planned switch: spends the player's turn and goes first
        sw = pending >= 0
        any_sw = sw.any()
        if any_sw:
            i = np.nonzero(sw)[0]
            if mech:
                cure = (F_pab[lb[i] * MAX_TEAM + p_idx[i]] & A["natural-cure"]) != 0
                p_status[i[cure], p_idx[i[cure]]] = NONE
            old = p_idx[i].copy()
            p_idx[i] = pending[i]
            switches[i] += 1
            reset_side(i, 0)
            if recording:
                for n_, k in enumerate(i):
                    rec(k, "switch", side="player", from_idx=int(old[n_]), to_idx=int(p_idx[k]),
                        from_id=specs[lb[k]].team[old[n_]], to_id=specs[lb[k]].team[p_idx[k]])
            entry_abilities(i, np.zeros(0, np.int64))
        pending = np.full(L, -1, np.int32)

        p_pos = lb * MAX_TEAM + p_idx
        r_pos = lb * MAX_TEAM + r_idx
        hpa = np.empty((L, 2), np.int32)
        hpa[:, 0] = p_hp.reshape(-1)[ar * MAX_TEAM + p_idx]
        hpa[:, 1] = r_hp.reshape(-1)[ar * MAX_TEAM + r_idx]
        maxa = np.empty((L, 2), np.int32)
        maxa[:, 0] = F_pmax[p_pos]
        maxa[:, 1] = F_rmax[r_pos]
        sta = np.empty((L, 2), np.int32)
        sta[:, 0] = p_status.reshape(-1)[ar * MAX_TEAM + p_idx]
        sta[:, 1] = r_status.reshape(-1)[ar * MAX_TEAM + r_idx]
        slp = np.empty((L, 2), np.int32)
        slp[:, 0] = p_sleep.reshape(-1)[ar * MAX_TEAM + p_idx]
        slp[:, 1] = r_sleep.reshape(-1)[ar * MAX_TEAM + r_idx]
        ab = np.stack([F_pab[p_pos], F_rab[r_pos]], 1)
        ty = np.stack([F_pty[p_pos], F_rty[r_pos]], 1)
        lvl = np.stack([F_plvl[p_pos], F_rlvl[r_pos]], 1)
        cur["hpa"], cur["sta"], cur["maxa"] = hpa, sta, maxa
        ab_box[0], ty_box[0], slp_box[0] = ab, ty, slp
        mu = ((lb * 6 + p_idx) * 6 + r_idx) * 2
        key = np.empty((L, 2), np.int32)
        key[:, 0] = S_keys[mu]
        key[:, 1] = S_keys[mu + 1]

        # move choice: policy table, then locked moves (charging, rampage)
        kk = np.empty((L, 2), np.int32)
        kk[:, 0] = POLICY[key[:, 0], np.minimum(hpa[:, 1], POLICY.shape[1] - 1)]
        kk[:, 1] = POLICY[key[:, 1], np.minimum(hpa[:, 0], POLICY.shape[1] - 1)]
        for q in (0, 1):
            for state, extra in ((charging, semi), (lock_mid, None)):
                ch = state[:, q] >= 0
                if state is lock_mid:
                    ch &= lock[:, q] > 0
                if ch.any():  # a locked move is used against whoever is on the field now
                    idx = np.nonzero(ch)[0]
                    same = A_midk[key[idx, q]] == state[idx, q][:, None]
                    found = same.any(1)
                    kk[idx[found], q] = same[found].argmax(1)
                    lost = idx[~found]
                    state[lost, q] = -1
                    if extra is not None:
                        extra[lost, q] = False
                    else:
                        lock[lost, q] = 0
        fkey = key * K + kk
        flags = A_flags[fkey]
        mids = A_mid[fkey]
        acts = np.ones((L, 2), bool)
        if any_sw:
            acts[:, 0] &= ~sw

        prio = A_prio[fkey]
        prio[recharge] = 0
        spe = np.stack([F_pspd[p_pos], F_rspd[r_pos]], 1)
        if mech:
            spe = spe * STAGE_NUM[stages[:, :, SPE] + 6] // STAGE_DEN[stages[:, :, SPE] + 6]
            spe = np.where(sta == PAR, spe // 4, spe)
        faster = spe[:, 0] > spe[:, 1]
        if not rules.tie_to_rival:
            faster |= (spe[:, 0] == spe[:, 1]) & ((R[:, S_HITS] & 1) == 1)  # uint16 is fine here
        p_first = (prio[:, 0] > prio[:, 1]) | ((prio[:, 0] == prio[:, 1]) & faster)

        ended = np.zeros(L, bool)
        rech_next = np.zeros((L, 2), bool)
        flinch = np.zeros((L, 2), bool)
        flinch_box[0] = flinch
        acc_stages = mech and bool(stages[:, :, ACC].any() or stages[:, :, EVA].any())
        hurt = np.zeros((L, 2), bool)  # hit by the opponent this turn (Revenge)
        for phase in (0, 1):
            a_side = np.where(p_first, phase, 1 - phase)
            i = np.nonzero(acts.reshape(-1)[ar * 2 + a_side] & ~ended)[0]
            if not len(i):
                continue
            aa = a_side[i]
            dd = 1 - aa
            ia0 = i * 2 + aa
            sta_f, slp_f, conf_f = sta.reshape(-1), slp.reshape(-1), confusion.reshape(-1)
            can = np.ones(len(i), bool)
            reason = np.full(len(i), "", dtype=object) if recording else None

            # ---- things that stop the move (generation 3 order)
            if mech:
                s_a = sta_f[ia0]
                asleep = s_a == SLP
                if asleep.any():
                    j = np.nonzero(asleep)[0]
                    dec = np.where((ab.reshape(-1)[ia0[j]] & A["early-bird"]) != 0, 2, 1)
                    slp_f[ia0[j]] -= dec
                    woke = slp_f[ia0[j]] <= 0
                    sta_f[ia0[j[woke]]] = NONE
                    slp_f[ia0[j[woke]]] = 0
                    can[j[~woke]] = False
                    if recording:
                        rec(i[j[woke]], "wake", side=aa[j[woke]])
                        reason[j[~woke]] = "sleep"
                frozen = (s_a == FRZ) & can
                if frozen.any():
                    j = np.nonzero(frozen)[0]
                    thaw = chance(i[j], aa[j], S_THAW, 20) | (E["special"][mids.reshape(-1)[ia0[j]]] == SP_THAW_USER)
                    sta_f[ia0[j[thaw]]] = NONE
                    can[j[~thaw]] = False
                    if recording:
                        rec(i[j[thaw]], "thaw", hpa_=hpa, sta_=sta, side=aa[j[thaw]])
                        reason[j[~thaw]] = "frozen"
            rch = recharge.reshape(-1)[ia0] & can
            can &= ~rch
            if recording:
                reason[rch] = "recharge"
            if mech:
                fl_ = flinch.reshape(-1)[ia0] & can
                can &= ~fl_
                if recording:
                    reason[fl_] = "flinch"
                conf = (conf_f[ia0] > 0) & can
                if conf.any():
                    j = np.nonzero(conf)[0]
                    conf_f[ia0[j]] -= 1
                    over = conf_f[ia0[j]] == 0
                    if recording:
                        rec(i[j[over]], "confusion_end", hpa_=hpa, sta_=sta, side=aa[j[over]])
                    still = j[~over]
                    if len(still):
                        hit_self = (rnd(i[still], aa[still], S_CONF) & 1) == 1
                        if recording:
                            rec(i[still[~hit_self]], "confused", hpa_=hpa, sta_=sta, side=aa[still[~hit_self]])
                        hs = still[hit_self]
                        if len(hs):
                            li, si = i[hs], aa[hs]
                            pos = lb[li] * MAX_TEAM + np.where(si == 0, p_idx[li], r_idx[li])
                            atk = np.where(si == 0, F_patk[pos], F_ratk[pos])
                            dfn = np.where(si == 0, F_pdef[pos], F_rdef[pos])
                            guts = ((ab[li, si] & A["guts"]) != 0) & (sta[li, si] != NONE)
                            atk = np.where(guts, atk * 3 // 2, atk)
                            atk = _stage(atk, stages[li, si, ATK])
                            dfn = _stage(dfn, stages[li, si, DEF])
                            dmg = ((2 * lvl[li, si] // 5 + 2) * atk * 40 // np.maximum(1, dfn)) // 50
                            dmg = np.where((sta[li, si] == BRN) & ~guts, dmg // 2, dmg)
                            dmg = np.maximum(1, dmg) + 2
                            hpa[li, si] = np.maximum(0, hpa[li, si] - dmg)
                            can[hs] = False
                            ended[li] |= hpa[li, si] == 0
                            if recording:
                                rec(li, "confusion_hit", hpa_=hpa, sta_=sta, side=si,
                                    damage=[_pct(d, mx) for d, mx in zip(dmg, maxa[li, si])], fainted=hpa[li, si] == 0)
                par = (sta_f[ia0] == PAR) & can
                if par.any():
                    j = np.nonzero(par)[0]
                    full = (rnd(i[j], aa[j], S_PARA) >> 14) == 0  # 1/4
                    can[j[full]] = False
                    if recording:
                        reason[j[full]] = "paralysis"
            stopped = ~can
            if stopped.any():  # an interrupted two-turn move or rampage ends
                si, sa_ = i[stopped], aa[stopped]
                charging[si, sa_] = -1
                semi[si, sa_] = False
                lock[si, sa_] = 0
                if recording:
                    for n_, k in enumerate(np.nonzero(stopped)[0]):
                        if reason[k]:
                            rec(i[k], "cant_move", hpa_=hpa, sta_=sta, side=int(aa[k]), reason=reason[k])
            i, aa, dd = i[can], aa[can], dd[can]
            if not len(i):
                continue

            # ---- start a two-turn move
            ia0 = i * 2 + aa
            fl = flags.reshape(-1)[ia0]
            starts = (charging.reshape(-1)[ia0] < 0) & ((fl & F_CHARGE) != 0)
            if starts.any():
                si, sa_ = i[starts], aa[starts]
                charging[si, sa_] = mids[si, sa_]
                semi[si, sa_] = (flags[si, sa_] & F_SEMI) != 0
            if starts.any() and recording:
                rec(si, "action", hpa_=hpa, sta_=sta, side=sa_, attacker=[who(k, s) for k, s in zip(si, sa_)],
                    defender=[who(k, 1 - s) for k, s in zip(si, sa_)],
                    move=[store.move_names[x] for x in mids[si, sa_]],
                    power=1, kind=[store.move_names[x] for x in mids[si, sa_]], damage=0, missed=False,
                    fainted=False, target_hp=0, crit=False, effectiveness=4, hits=0)
            i, aa, dd = i[~starts], aa[~starts], dd[~starts]
            if not len(i):
                continue

            # ---- use the move
            ia = i * 2 + aa  # flat (lane, side) of the attacker
            idf = ia ^ 1
            fi = fkey.reshape(-1)[ia]
            mv = mids.reshape(-1)[ia]
            fl = flags.reshape(-1)[ia]
            target_semi = semi.reshape(-1)[idf]
            target_charge = charging.reshape(-1)[idf]
            charging.reshape(-1)[ia] = -1
            semi.reshape(-1)[ia] = False
            h_att = hpa.reshape(-1)[ia]
            h_def = hpa.reshape(-1)[idf]
            ab_a, ab_d = ab.reshape(-1)[ia], ab.reshape(-1)[idf]
            sta_a, sta_d = sta.reshape(-1)[ia], sta.reshape(-1)[idf]
            n_i = len(i)
            r_acc = rnd(i, aa, S_ACC)

            # accuracy
            acc = A_acc[fi]
            failed = np.zeros(n_i, bool)
            if mech:
                damp = ((fl & F_SELF_KO) != 0) & (((ab_a | ab_d) & A["damp"]) != 0)
                failed |= damp
                semi_ok = np.zeros(n_i, bool)
                if target_semi.any():
                    kind = E["semi_kind"][np.maximum(target_charge, 0)]
                    semi_ok = target_semi & (E["hits_semi"][mv] == kind) & (kind > 0)
                blocked = target_semi & ~semi_ok
                if acc_stages:
                    stg_f = stages.reshape(-1)
                    buff = np.clip(stg_f[ia * 7 + ACC] - stg_f[idf * 7 + EVA], -6, 6) + 6
                    calc_acc = acc * ACC_NUM[buff] // ACC_DEN[buff]
                else:
                    calc_acc = acc
                calc_acc = np.where((ab_a & A["compound-eyes"]) != 0, calc_acc * 130 // 100, calc_acc)
                hit = (acc >= NEVER_MISS) | (((r_acc * 100) >> 16) < calc_acc)
                ohko = (fl & F_OHKO) != 0
                if ohko.any():
                    la, ld = lvl.reshape(-1)[ia], lvl.reshape(-1)[idf]
                    ohko_hit = (la >= ld) & (((r_acc * 100) >> 16) < (30 + la - ld)) & ((ab_d & A["sturdy"]) == 0)
                    hit = np.where(ohko, ohko_hit, hit)
                hit &= ~blocked & ~failed
            else:
                semi_ok = np.zeros(n_i, bool)
                hit = ((acc >= NEVER_MISS) | (((r_acc * 100) >> 16) < acc)) & ~target_semi

            # damage of one hit
            crit = rnd(i, aa, S_CRIT) < A_thr[fi]
            phys = (fl & F_PHYS) != 0
            fixed = (fl & F_FIXED) != 0
            table = ((fl & F_TABLE) != 0) | fixed | ((fl & (F_HALF_HP | F_ENDEAVOR | F_OHKO)) != 0)
            base = np.where(crit, A_crit[fi], A_base[fi])
            if mech:
                # At stage 0, without status or special power, the formula equals the calculator
                # (checked for every matchup): recompute only where something changes it.
                stg_f = stages.reshape(-1)
                sa_raw = np.where(phys, stg_f[ia * 7 + ATK], stg_f[ia * 7 + SPA])
                sd_raw = np.where(phys, stg_f[idf * 7 + DEF], stg_f[idf * 7 + SPD])
                special = E["special"][mv]
                magnitude = special == SP_MAGNITUDE
                needs = ((sa_raw != 0) | (sd_raw != 0) | (sta_a != NONE) | (special != 0) | semi_ok
                         | ((fl & F_PINCH) != 0)) & (~table | magnitude)
                burned_tab = table & phys & ~fixed & (sta_a == BRN) & ((ab_a & A["guts"]) == 0)
                if burned_tab.any():  # damage from the calculator: approximate the burn
                    base = np.where(burned_tab, base // 2, base)
                j = np.nonzero(needs)[0]
                if len(j):
                    fj, pj, cj, sa_j, st_j = fi[j], phys[j], crit[j], sta_a[j], special[j]
                    guts = ((ab_a[j] & A["guts"]) != 0) & (sa_j != NONE)
                    at = np.where(guts & pj, A_at[fj] * 3 // 2, A_at[fj])
                    at = _stage(at, np.where(cj, np.maximum(sa_raw[j], 0), sa_raw[j]))
                    df = np.maximum(1, _stage(A_df[fj], np.where(cj, np.minimum(sd_raw[j], 0), sd_raw[j])))
                    bp = A_bp[fj]
                    ha, mxa = h_att[j], maxa.reshape(-1)[ia[j]]
                    bp = np.where(((fl[j] & F_PINCH) != 0) & (ha <= mxa // 3), bp * 150 // 100, bp)
                    bp = np.where((st_j == SP_FACADE) & np.isin(sa_j, (PAR, BRN, PSN, TOX)), bp * 2, bp)
                    bp = np.where((st_j == SP_SMELLING_SALT) & (sta_d[j] == PAR), bp * 2, bp)
                    bp = np.where((st_j == SP_REVENGE) & hurt.reshape(-1)[ia[j]], bp * 2, bp)
                    if (st_j == SP_FLAIL).any():
                        p48 = 48 * ha // np.maximum(1, mxa)
                        flail = np.select([p48 <= 1, p48 <= 4, p48 <= 9, p48 <= 16, p48 <= 32],
                                          [200, 150, 100, 80, 40], 20)
                        bp = np.where(st_j == SP_FLAIL, flail, bp)
                    mj = st_j == SP_MAGNITUDE
                    if mj.any():
                        u = (rnd(i[j], aa[j], S_MAGNITUDE) * 100) >> 16
                        bp = np.where(mj, MAGNITUDE_BP[np.searchsorted(MAGNITUDE_CUM, u, side="right")], bp)
                    bp = np.where(semi_ok[j] & (E["double_semi"][mv[j]] == 1), bp * 2, bp)
                    burned = (sa_j == BRN) & ~guts
                    dmg = ((2 * A_level[fj // K] // 5 + 2) * at * bp // df) // 50
                    dmg = np.where(pj & burned, dmg // 2, dmg)
                    dmg = np.where(pj, np.maximum(1, dmg), dmg) + 2
                    dmg = np.where(cj, dmg * 2, dmg)
                    dmg = np.where((fl[j] & F_STAB) != 0, dmg * 3 // 2, dmg)
                    eff4 = A_eff4[fj]
                    base[j] = np.where(eff4 > 0, np.maximum(1, dmg * eff4 // 4), 0)  # at least 1 unless immune
            roll = rnd(i, aa, S_ROLL) >> 12
            total = np.where(fixed, base, np.where(base > 0, np.maximum(1, (base * (85 + roll)) // 100), 0))
            hk = A_hits[fi]
            nh = np.ones(n_i, np.int32)
            multi = np.nonzero(hk > 1)[0]
            if len(multi):  # extra hits, each with its own roll
                nh[multi] = np.where(hk[multi] == 5, HITS_2_5[rnd(i[multi], aa[multi], S_HITS) >> 13], hk[multi])
                bm, fm = base[multi], fixed[multi]
                for h in range(1, 5):
                    roll = rnd(i[multi], aa[multi], S_ROLL + h) >> 12
                    dmg_h = np.where(fm, bm, np.where(bm > 0, np.maximum(1, (bm * (85 + roll)) // 100), 0))
                    total[multi] += np.where(h < nh[multi], dmg_h, 0)
            if (fl & (F_HALF_HP | F_ENDEAVOR | F_NO_KO | F_OHKO)).any():
                total = np.where(fl & F_HALF_HP, np.maximum(1, h_def // 2), total)
                total = np.where(fl & F_ENDEAVOR, np.maximum(0, h_def - h_att), total)
                total = np.where(fl & F_OHKO, h_def, total)
                cap = np.where(fl & F_NO_KO, h_def - 1, h_def)
            else:
                cap = h_def
            real = np.where(hit, np.minimum(total, cap), 0)
            h_def = h_def - real
            drain = A_drain[fi]
            if drain.any():
                h_att = np.minimum(maxa.reshape(-1)[ia], h_att + np.where(drain > 0, (real * drain) // 100, 0))
                recoil = np.where((drain < 0) & (real > 0), np.maximum(1, (real * -drain) // 100), 0)
                if mech:
                    recoil = np.where((ab_a & A["rock-head"]) != 0, 0, recoil)
                h_att = h_att - recoil
            if (fl & (F_CRASH | F_SELF_KO)).any():
                h_att = h_att - np.where(((fl & F_CRASH) != 0) & ~hit & ~failed, total // 2, 0)
                h_att = np.where(((fl & F_SELF_KO) != 0) & ~failed, 0, h_att)
            h_att = np.maximum(0, h_att)
            hpa.reshape(-1)[idf] = h_def
            hpa.reshape(-1)[ia] = h_att
            hurt.reshape(-1)[idf] |= real > 0
            rech = ((fl & F_RECHARGE) != 0) & (h_att > 0)
            if not rules.recharge_on_miss:
                rech &= hit
            if rech.any():
                rech_next.reshape(-1)[ia[rech]] = True
            if recording:
                rec(i, "action", hpa_=hpa, sta_=sta, side=aa, attacker=[who(k, s) for k, s in zip(i, aa)],
                    defender=[who(k, 1 - s) for k, s in zip(i, aa)], move=[store.move_names[x] for x in mv],
                    power=A_eff4[fi] if mech else np.where(base > 0, 1, 0), kind="attack",
                    damage=[_pct(d, mx) for d, mx in zip(real, maxa.reshape(-1)[idf])],
                    missed=~hit, failed=failed, crit=crit & hit & (real > 0), effectiveness=A_eff4[fi],
                    hits=np.where(hit, nh, 0), fainted=h_def == 0, target_hp=[_pct(h, mx) for h, mx in
                                                                            zip(h_def, maxa.reshape(-1)[idf])])
                heal = np.nonzero((drain > 0) & (real > 0))[0]
                if len(heal):
                    rec(i[heal], "heal", hpa_=hpa, sta_=sta, side=aa[heal],
                        amount=[_pct(x, mx) for x, mx in zip((real[heal] * drain[heal]) // 100,
                                                            maxa.reshape(-1)[ia[heal]])])
                rc = np.nonzero((drain < 0) & (real > 0))[0]
                if len(rc):
                    rec(i[rc], "recoil", hpa_=hpa, sta_=sta, side=aa[rc])

            if mech:
                effects_after_hit(i, aa, dd, mv, fl, hit, real, h_att, h_def, ab_a, ab_d, phase)

            # rampage (Thrash, Petal Dance, Outrage): 2-3 turns, then the user is confused
            if mech and ((fl & F_LOCK) != 0).any():
                j = np.nonzero((fl & F_LOCK) != 0)[0]
                li, si = i[j], aa[j]
                start = lock[li, si] == 0
                lock[li[start], si[start]] = 2 + ((rnd(li[start], si[start], S_DUR) >> 15) & 1)
                lock_mid[li, si] = mv[j]
                lock[li, si] -= 1
                done_ = (lock[li, si] == 0) & (hpa[li, si] > 0)
                if done_.any():
                    cl, cs = li[done_], si[done_]
                    okc = (confusion[cl, cs] == 0) & ((ab[cl, cs] & A["own-tempo"]) == 0)
                    confusion[cl[okc], cs[okc]] = 2 + ((rnd(cl[okc], cs[okc], S_DUR) >> 13) & 3)
                    if recording:
                        rec(cl[okc], "confuse", side=cs[okc], cause="fatigue")
            ended[i] |= (hpa[i, 0] == 0) | (hpa[i, 1] == 0)

        if rules.charge_while_recharging:
            quirk = recharge & (charging < 0) & ((flags & F_CHARGE) != 0) & (hpa > 0)
            if quirk.any():
                qi, qs = np.nonzero(quirk)
                charging[qi, qs] = mids[qi, qs]
                semi[qi, qs] = (flags[qi, qs] & F_SEMI) != 0
        recharge = rech_next

        # ---- end of turn: Shed Skin, poison, burn, trapping moves
        if mech:
            end_of_turn(ar)

        p_hp.reshape(-1)[ar * MAX_TEAM + p_idx] = hpa[:, 0]
        r_hp.reshape(-1)[ar * MAX_TEAM + r_idx] = hpa[:, 1]
        p_status.reshape(-1)[ar * MAX_TEAM + p_idx] = sta[:, 0]
        r_status.reshape(-1)[ar * MAX_TEAM + r_idx] = sta[:, 1]
        p_sleep.reshape(-1)[ar * MAX_TEAM + p_idx] = slp[:, 0]
        r_sleep.reshape(-1)[ar * MAX_TEAM + r_idx] = slp[:, 1]
        cur["hpa"] = cur["sta"] = None

        # faints
        r_f = hpa[:, 1] == 0
        p_f = hpa[:, 0] == 0
        if r_f.any() or p_f.any():
            r_idx = r_idx + r_f
            rf = np.nonzero(r_f)[0]
            reset_side(rf, 1)
            p_alive = p_alive - p_f
            rival_done = r_idx >= S_m[lb]
            changed = np.nonzero((r_f | p_f) & ~rival_done & (p_alive > 0))[0]
            new_p = np.zeros(0, np.int64)
            if len(changed):
                c_plan = S_plan[lb[changed], r_idx[changed]]
                c_alive = (c_plan >= 0) & (p_hp[changed, np.maximum(c_plan, 0)] > 0)
                need = p_f[changed]
                if need.any():  # free replacement: the planned member, otherwise the next alive in order
                    nidx = changed[need]
                    first_alive = (p_hp[nidx] > 0).argmax(1)
                    p_idx[nidx] = np.where(c_alive[need], c_plan[need], first_alive)
                    reset_side(nidx, 0)
                    new_p = nidx
                    if recording:
                        for k in nidx:
                            rec(k, "send_out", side="player", idx=int(p_idx[k]))
                ask = r_f[changed] & c_alive & (c_plan != p_idx[changed])
                pending[changed[ask]] = c_plan[ask]
            new_r = rf[~rival_done[rf] & (p_alive[rf] > 0)]
            if recording:
                for k in new_r:
                    rec(k, "send_out", side="rival", idx=int(r_idx[k]))
            entry_abilities(new_p, new_r)
            done = rival_done | (p_alive == 0)
            if done.any():
                d = np.nonzero(done)[0]
                won = rival_done[d] & (p_alive[d] > 0)
                lost = _store_results(out, b0, lb[d], ls[d], p_hp[d], S_pmax[lb[d]], r_hp[d], S_rmax[lb[d]],
                                      switches[d], turn, won, np.zeros(len(d), bool))
                if recording:
                    for n_, k in enumerate(d):
                        rec(k, "end", won=bool(won[n_]), timed_out=False, hp_lost=int(round(lost[n_])))
                keep = ~done
                lb, ls, rrow = lb[keep], ls[keep], rrow[keep]
                p_hp, r_hp = p_hp[keep], r_hp[keep]
                p_alive, p_idx, r_idx = p_alive[keep], p_idx[keep], r_idx[keep]
                p_status, r_status, p_sleep, r_sleep = p_status[keep], r_status[keep], p_sleep[keep], r_sleep[keep]
                charging, semi, recharge, stages = charging[keep], semi[keep], recharge[keep], stages[keep]
                confusion, trap, lock, lock_mid, tox_n = confusion[keep], trap[keep], lock[keep], lock_mid[keep], tox_n[keep]
                switches, pending = switches[keep], pending[keep]
                L = len(lb)
        continue


def ka_(fi):
    """Key id of a flat (key * K + candidate) index."""
    return fi // K


def _store_results(out, b0, lb, ls, p_hp, p_max, r_hp, r_max, switches, turns, won, timed_out):
    with np.errstate(invalid="ignore", divide="ignore"):
        lost = np.where(p_max > 0, (p_max - p_hp) * 100.0 / np.maximum(p_max, 1), 0.0).sum(1)
        left = np.where(r_max > 0, r_hp * 100.0 / np.maximum(r_max, 1), 0.0).sum(1)
    b = b0 + lb
    out.won[b, ls] = won
    out.hp_lost[b, ls] = lost
    out.rival_hp_left[b, ls] = left
    out.turns[b, ls] = turns
    out.switches[b, ls] = switches
    out.timed_out[b, ls] = timed_out
    return lost


# --------------------------------------------------------------------------- whole individuals


@dataclass
class GenomeScores:
    """Per genome: fitness of every scenario (G, n), win rate of every battle (G, 21) and the
    probability of winning all 21 battles in the same scenario (G,)."""
    fitness: np.ndarray
    win_rates: np.ndarray
    p_complete: np.ndarray
    battle_hp_lost: np.ndarray = None  # (G, 21) mean % HP lost in every battle
    battle_turns: np.ndarray = None  # (G, 21) mean turns
    battle_switches: np.ndarray = None  # (G, 21) mean switches
    hp_lost: np.ndarray = None  # (G, n) total % HP lost in every scenario

    @property
    def mean(self) -> np.ndarray:
        return self.fitness.mean(1)

    @property
    def stderr(self) -> np.ndarray:
        return self.fitness.std(1, ddof=1) / np.sqrt(self.fitness.shape[1])


def score_genomes(genomes: list, n: int, seed: int = 0, store: MoveStore | None = None, chunk: int = 2000,
                  start: int = 0) -> GenomeScores:
    """Plays the 21 battles of every genome in ``n`` scenarios (from ``start``), ``chunk``
    scenarios at a time so that large sets fit in memory."""
    store = store or default_store()
    specs = [compile_battle(g, t, store) for g in genomes for t in range(N_TRAINERS)]
    G = len(genomes)
    fitness, hp_total = np.zeros((G, n)), np.zeros((G, n))
    wins, hp, turns, sw = (np.zeros((G, N_TRAINERS)) for _ in range(4))
    complete = np.zeros(G)
    for s0 in range(0, n, chunk):
        sc = Scenarios(min(chunk, n - s0), seed=seed, start=start + s0)
        res = simulate(specs, sc, store)
        shape = (G, N_TRAINERS, -1)
        cost = np.where(res.won, res.hp_lost, bs.LOSS_PENALTY).reshape(shape)
        won = res.won.reshape(shape)
        fitness[:, s0:s0 + sc.n] = cost.sum(1)
        hp_total[:, s0:s0 + sc.n] = res.hp_lost.reshape(shape).sum(1)
        wins += won.sum(2)
        hp += res.hp_lost.reshape(shape).sum(2)
        turns += res.turns.reshape(shape).sum(2)
        sw += res.switches.reshape(shape).sum(2)
        complete += won.all(1).sum(1)
    return GenomeScores(fitness, wins / n, complete / n, hp / n, turns / n, sw / n, hp_total)


def replay(genome, scenario: int, seed: int = 0, store: MoveStore | None = None) -> list:
    """The 21 battles of ``genome`` in one scenario, with every event (for the viewer). Returns
    ``battle_simulator.BattleResult`` objects."""
    store = store or default_store()
    specs = [compile_battle(genome, t, store) for t in range(N_TRAINERS)]
    res = simulate(specs, Scenarios(1, seed=seed, start=scenario), store, record=True)
    out = []
    for t, spec in enumerate(specs):
        out.append(bs.BattleResult(
            won=bool(res.won[t, 0]), hp_lost=int(round(float(res.hp_lost[t, 0]))),
            rival_hp_left=int(round(float(res.rival_hp_left[t, 0]))), turns=int(res.turns[t, 0]),
            timed_out=bool(res.timed_out[t, 0]), level_cap=spec.level_cap, team=spec.team, rival_team=spec.rival_team,
            switches=int(res.switches[t, 0]), events=res.events[t][0]))
    return out


def explain_choice(store: MoveStore, key_id: int, hp: int) -> tuple:
    """``(move, reason)``: move the policy chooses against a defender with ``hp`` HP left and why
    ("debilita seguro": its lowest roll KOs; "más daño esperado": otherwise)."""
    hp = int(min(max(hp, 1), store["policy"].shape[1] - 1))
    k = int(store["policy"][key_id, hp])
    move = store.move_names[int(store["move_id"][key_id, k])]
    sure = int(store["ko_min"][key_id, k]) >= hp
    return move, ("debilita seguro" if sure else "más daño esperado")


# --------------------------------------------------------------------------- data


_STORE = None


def default_store() -> MoveStore:
    """Move store built from pkm_data/damage_rolls.json (loaded once per process)."""
    global _STORE
    if _STORE is None:
        from ga.damage_table import load_rolls

        rolls = load_rolls()
        if not rolls:
            raise FileNotFoundError("No existe pkm_data/damage_rolls.json (o es de una versión anterior): arranca el "
                                    "servidor de daños y ejecuta 'python tools/precompute_damage.py --rolls'")
        _STORE = MoveStore.from_rolls(rolls, bs.dataset["moves"])
    return _STORE
