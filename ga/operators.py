"""Genetic operators for the nuzlocke individual.

Genome: ``[catches, teams, plans]``
  * ``catches``: 36 pkm ids (or ``None``), one per route in ``ROUTES_ORDER``.
  * ``teams``: 21 lists of up to 6 pkm ids in send-out order, one per trainer in
    ``TRAINERS_ORDER``. Teams 16..20 (Pokémon League) must contain the same
    Pokémon, possibly in a different order.
  * ``plans``: switch plan of each battle, one entry per rival Pokémon (``RIVAL_SIZES[t]``)
    with the team member (pkm id) that must face it, or ``None`` to keep the Pokémon on
    the field. When that rival Pokémon comes out, the player spends a turn switching to the
    planned member; when a Pokémon faints, the planned member is sent out (free).
    A genome without plans (``[catches, teams]``, or all ``None``) never switches.

Constraints (``violations`` / ``is_feasible``): no species caught twice, every team only
uses Pokémon caught before that battle, at most 6 per team, the league teams share their
members and every plan only names members of its team.
"""
from __future__ import annotations

import copy
import random

from route_data.routes import ROUTES, ROUTES_ORDER
from route_data.trainers import PREVIOUS_ROUTES_TO_TRAINER, TRAINERS, TRAINERS_ORDER

N_ROUTES = len(ROUTES_ORDER)
N_TRAINERS = len(TRAINERS_ORDER)
LEAGUE_FIRST = TRAINERS_ORDER.index("ALTO_MANDO_LORELEI")  # 16
LEAGUE = tuple(range(LEAGUE_FIRST, N_TRAINERS))  # 16..20
TEAM_SIZE = 6
# Routes with index < CUT[t] have been visited before battling trainer t
CUT = tuple(PREVIOUS_ROUTES_TO_TRAINER[name] for name in TRAINERS_ORDER)
ROUTE_OPTIONS = tuple(tuple(dict.fromkeys(ROUTES[r])) for r in ROUTES_ORDER)  # unique species per route


def _rival_size(name: str) -> int:
    trainer = TRAINERS[name]
    return len(next(iter(trainer.values()))) if isinstance(trainer, dict) else len(trainer)


# Rival team size of each battle (the same for the three starters)
RIVAL_SIZES = tuple(_rival_size(name) for name in TRAINERS_ORDER)

# Heuristic for smart initialisation: preferred / avoided types against each gym leader
TYPE_PREFERENCES = {
    1: ["water", "grass", "fighting"],  # GYM PLATEADA (rock)
    3: ["electric", "grass", "bug"],  # GYM CELESTE (water)
    5: ["ground", "rock"],  # GYM CARMIN (electric)
    7: ["psychic", "poison", "fire", "flying"],  # GYM AZULONA (grass)
    11: ["psychic"],  # GYM FUCSIA (poison)
    12: ["bug", "ghost"],  # GYM AZAFRAN (psychic)
    13: ["water", "ground", "rock"],  # GYM CANELA (fire)
    14: ["water", "grass", "ground"],  # GYM VERDE (ground)
}
TYPE_NOT_PREFERENCES = {
    1: ["fire", "bug", "flying"],
    3: ["rock", "fire", "ground"],
    5: ["flying", "water"],
    7: ["water", "ground", "rock"],
    11: ["grass"],
    12: ["poison"],
    13: ["grass", "bug"],
    14: ["fire"],
}


# --------------------------------------------------------------------------- helpers


def empty_plans() -> list:
    return [[None] * size for size in RIVAL_SIZES]


def normalize(genome) -> list:
    """Returns ``[catches, teams, plans]`` (adds empty plans to old two-part genomes)."""
    catches, teams = list(genome[0]), [list(t) for t in genome[1]]
    plans = genome[2] if len(genome) > 2 and genome[2] is not None else None
    if plans is None:
        plans = empty_plans()
    else:
        plans = [(list(p) + [None] * RIVAL_SIZES[t])[:RIVAL_SIZES[t]] for t, p in enumerate(plans)]
    return [catches, teams, plans]


def genome_key(genome) -> tuple:
    plans = genome[2] if len(genome) > 2 else None
    plan_key = tuple(tuple(p) for p in plans) if plans and any(x is not None for p in plans for x in p) else ()
    return tuple(genome[0]), tuple(tuple(t) for t in genome[1]), plan_key


def has_switching(genome) -> bool:
    return len(genome) > 2 and any(x is not None for p in genome[2] for x in p)


def available_for(catches, trainer_index: int) -> list:
    return [p for p in catches[:CUT[trainer_index]] if p is not None]


def dedupe(seq) -> list:
    """Removes duplicates keeping the first occurrence (order matters: it is the send-out order)."""
    return list(dict.fromkeys(seq))


def league_permutations(teams) -> list:
    """For each league team after the first, the positions of its members in team 16."""
    base = teams[LEAGUE_FIRST]
    pos = {p: i for i, p in enumerate(base)}
    return [[pos[p] for p in teams[t] if p in pos] for t in LEAGUE[1:]]


def derive_league(team16: list, perms: list) -> list:
    """League teams 17..20 as the given permutations of ``team16`` (robust to size changes)."""
    out = []
    for perm in perms:
        seq = [team16[i] for i in perm if i < len(team16)]
        seq = dedupe(seq)
        seq += [p for p in team16 if p not in seq]
        out.append(seq)
    return out


def remap_plans(plans: list, old_teams: list, new_teams: list) -> None:
    """Keeps the plans valid after the teams change (in place): a planned member that left its
    team is replaced by the Pokémon that took its slot, or dropped."""
    for t, plan in enumerate(plans):
        new = new_teams[t]
        members = set(new)
        old = old_teams[t]
        for j, pkm in enumerate(plan):
            if pkm is None or pkm in members:
                continue
            slot = old.index(pkm) if pkm in old else None
            plan[j] = new[slot] if slot is not None and slot < len(new) else None


def _weighted_sample(rng: random.Random, items: list, weights: list, k: int) -> list:
    """k distinct items, each pick proportional to its weight, keeping pick order."""
    items, weights = list(items), list(weights)
    chosen = []
    for _ in range(min(k, len(items))):
        i = rng.choices(range(len(items)), weights=weights, k=1)[0]
        chosen.append(items.pop(i))
        weights.pop(i)
    return chosen


def _type_weight(pkm_id: int, trainer_index: int, pkdex: dict) -> float:
    weight = 1.0
    for typing in pkdex[str(pkm_id)]["types"]:
        if typing in TYPE_PREFERENCES.get(trainer_index, []):
            weight *= 2
        if typing in TYPE_NOT_PREFERENCES.get(trainer_index, []):
            weight /= 2
    return weight


def best_matchup(team: list, trainer_index: int, rival_index: int, starter, dataset=None):
    """Team member with the best exchange against a rival Pokémon according to the damage table:
    damage it deals minus damage it receives (both as fractions of HP)."""
    import battle_simulator as bs
    from route_data.evolve_obj import AVAILABLE_EVOLVE_OBJ_TRAINERS

    dataset = dataset if dataset is not None else bs.dataset
    rival = bs.get_rival_team(trainer_index, starter)[rival_index]
    cap = bs.select_level_cap(bs.get_rival_team(trainer_index, starter))
    items = AVAILABLE_EVOLVE_OBJ_TRAINERS[TRAINERS_ORDER[trainer_index]]
    table = dataset["damage_dictionary"]
    best, best_score = None, float("-inf")
    for pkm in team:
        name = dataset["pkdex"][str(bs.evolve_pokemon(pkm, cap, items, dataset))]["name"]
        dealt = table.get(bs.damage_key(name, cap, rival["species"], rival["level"]), [None, 0.0])[1]
        taken = table.get(bs.damage_key(rival["species"], rival["level"], name, cap, True), [None, 0.0])[1]
        if dealt - taken > best_score:
            best, best_score = pkm, dealt - taken
    return best


def random_plan_entry(rng: random.Random, team: list, trainer_index: int, rival_index: int, starter,
                      smart: bool):
    if not team:
        return None
    if smart and rng.random() < 0.6:
        return best_matchup(team, trainer_index, rival_index, starter)
    return rng.choice(team)


# --------------------------------------------------------------------------- generation


def generate_individual(rng: random.Random | None = None, smart: bool = False, pkdex: dict | None = None,
                        switching: bool = False, plan_prob: float = 0.35) -> list:
    """Random feasible individual. ``smart`` weights gym teams (and plans) by matchups; with
    ``switching`` each rival Pokémon gets a planned opponent with probability ``plan_prob``."""
    rng = rng or random
    used = set()
    catches = []
    for options in ROUTE_OPTIONS:
        available = [p for p in options if p not in used]
        chosen = rng.choice(available) if available else None
        if chosen is not None:
            used.add(chosen)
        catches.append(chosen)

    if smart and pkdex is None:
        from battle_simulator import dataset
        pkdex = dataset["pkdex"]

    teams = []
    for t in range(LEAGUE_FIRST + 1):
        available = available_for(catches, t)
        k = min(TEAM_SIZE, len(available))
        if smart and t in TYPE_PREFERENCES:
            teams.append(_weighted_sample(rng, available, [_type_weight(p, t, pkdex) for p in available], k))
        else:
            teams.append(rng.sample(available, k))
    for _ in LEAGUE[1:]:
        teams.append(rng.sample(teams[LEAGUE_FIRST], len(teams[LEAGUE_FIRST])))

    plans = empty_plans()
    if switching:
        for t in range(N_TRAINERS):
            for j in range(RIVAL_SIZES[t]):
                if rng.random() < plan_prob:
                    plans[t][j] = random_plan_entry(rng, teams[t], t, j, catches[0], smart)
    return [catches, teams, plans]


# --------------------------------------------------------------------------- repair


def repair_teams(catches: list, teams: list, rng: random.Random, perms: list | None = None) -> None:
    """Makes teams 0..16 use only available Pokémon (in place, keeping positions),
    fills them up to min(6, available) and derives the league teams from team 16."""
    if perms is None:
        perms = league_permutations(teams)
    for t in range(LEAGUE_FIRST + 1):
        available = available_for(catches, t)
        avail_set = set(available)
        team = []
        for p in teams[t][:TEAM_SIZE]:
            if p in avail_set and p not in team:
                team.append(p)
            else:
                options = [x for x in available if x not in team and x not in teams[t]]
                if options:
                    team.append(rng.choice(options))
        need = min(TEAM_SIZE, len(available)) - len(team)
        if need > 0:
            team += rng.sample([x for x in available if x not in team], need)
        teams[t] = team
    teams[LEAGUE_FIRST + 1:] = derive_league(teams[LEAGUE_FIRST], perms)


# --------------------------------------------------------------------------- mutation


def mutate(genome: list, rng: random.Random, p_catch: float = 0.15, p_team: float = 0.15,
           p_order: float = 0.1, p_plan: float = 0.0) -> list:
    """Four-level mutation (returns a new genome):

    1. each capture is replaced, with probability ``p_catch``, by another species of
       the same route; a species caught later can be moved earlier (its old route is
       re-rolled). Teams that used the replaced Pokémon get the new one in the same slot.
    2. each team slot is replaced, with probability ``p_team``, by another available
       Pokémon not in the team.
    3. each team swaps two positions with probability ``p_order`` (send-out order).
    4. each plan entry changes, with probability ``p_plan``, to another team member or to
       "keep" (``None``). Plans always follow the team changes of steps 1-2.
    """
    catches, teams, plans = copy.deepcopy(normalize(genome))
    original_teams = [list(t) for t in teams]
    perms = league_permutations(teams)

    # 1. captures
    for i in range(N_ROUTES):
        if rng.random() >= p_catch:
            continue
        earlier = set(catches[:i])
        options = [p for p in ROUTE_OPTIONS[i] if p != catches[i] and p not in earlier]
        if not options:
            continue
        old, new = catches[i], rng.choice(options)
        catches[i] = new
        if new in catches[i + 1:]:
            j = catches.index(new, i + 1)
            taken = set(catches)
            reroll = [p for p in ROUTE_OPTIONS[j] if p not in taken]
            catches[j] = rng.choice(reroll) if reroll else None
        for t in range(LEAGUE_FIRST + 1):
            if i < CUT[t]:
                teams[t] = [new if p == old else p for p in teams[t]]
    repair_teams(catches, teams, rng, perms)

    # 2. team members
    for t in range(LEAGUE_FIRST + 1):
        available = available_for(catches, t)
        for pos in range(len(teams[t])):
            if rng.random() < p_team:
                options = [p for p in available if p not in teams[t]]
                if options:
                    teams[t][pos] = rng.choice(options)
    teams[LEAGUE_FIRST + 1:] = derive_league(teams[LEAGUE_FIRST], perms)
    remap_plans(plans, original_teams, teams)

    # 3. send-out order
    for t in range(N_TRAINERS):
        if len(teams[t]) >= 2 and rng.random() < p_order:
            a, b = rng.sample(range(len(teams[t])), 2)
            teams[t][a], teams[t][b] = teams[t][b], teams[t][a]

    # 4. switch plans
    if p_plan > 0:
        for t in range(N_TRAINERS):
            for j in range(RIVAL_SIZES[t]):
                if rng.random() < p_plan:
                    options = [p for p in teams[t] if p != plans[t][j]]
                    plans[t][j] = None if (plans[t][j] is not None and rng.random() < 0.35) or not options \
                        else rng.choice(options)
    return [catches, teams, plans]


# --------------------------------------------------------------------------- crossover


def _assign_required(required: dict, preferred: dict, rng: random.Random) -> dict:
    """Bipartite matching Pokémon -> route with deadlines (augmenting paths).

    ``required[p]`` is the exclusive route deadline for Pokémon ``p`` (it must be
    caught at a route index < deadline). Routes where a parent caught ``p`` are
    tried first. Returns ``route -> pkm`` for the Pokémon that could be placed.
    """
    candidates = {}
    for p, deadline in required.items():
        valid = [r for r in range(min(deadline, N_ROUTES)) if p in ROUTE_OPTIONS[r]]
        pref = [r for r in preferred.get(p, []) if r in valid]
        rest = [r for r in valid if r not in pref]
        candidates[p] = pref + rest
    route_of: dict[int, int] = {}  # route -> pkm

    def try_assign(p, seen):
        for r in candidates[p]:
            if r in seen:
                continue
            seen.add(r)
            if r not in route_of or try_assign(route_of[r], seen):
                route_of[r] = p
                return True
        return False

    order = sorted(required, key=lambda p: (len(candidates[p]), required[p], rng.random()))
    for p in order:
        try_assign(p, set())
    return route_of


def crossover(parent1: list, parent2: list, rng: random.Random) -> list:
    """Team-level uniform crossover with feasible capture reconstruction.

    The child takes each of the first 16 battles (team and switch plan) from a random parent
    (half and half) and the whole league block (teams and plans 16..20, keeping the send-out
    permutations) from one parent. Captures are inherited route by route; the Pokémon the
    teams need are placed with a deadline-aware matching that prefers the routes where a
    parent caught them. Pokémon that cannot be placed are replaced in their team (same slot)
    and the plans follow. The child never has repeated captures.
    """
    parent1, parent2 = normalize(parent1), normalize(parent2)
    league_parent = parent1 if rng.random() < 0.5 else parent2
    idx = list(range(LEAGUE_FIRST))
    rng.shuffle(idx)
    from_p1 = set(idx[:LEAGUE_FIRST // 2])
    sources = [parent1 if t in from_p1 else parent2 for t in range(LEAGUE_FIRST)] + [league_parent] * len(LEAGUE)
    teams = [list(src[1][t]) for t, src in enumerate(sources)]
    plans = [list(src[2][t]) for t, src in enumerate(sources)]
    perms = league_permutations(league_parent[1])

    required: dict[int, int] = {}
    for t in range(LEAGUE_FIRST + 1):
        for p in teams[t]:
            required[p] = min(required.get(p, N_ROUTES), CUT[t])
    preferred: dict[int, list] = {}
    for parent in (parent1, parent2):
        for r, p in enumerate(parent[0]):
            if p is not None:
                preferred.setdefault(p, []).append(r)
    route_of = _assign_required(required, preferred, rng)

    catches = [route_of.get(r) for r in range(N_ROUTES)]
    caught = set(route_of.values())
    parents = (parent1, parent2) if rng.random() < 0.5 else (parent2, parent1)
    for r in range(N_ROUTES):
        if catches[r] is not None:
            continue
        choice = next((par[0][r] for par in parents if par[0][r] is not None and par[0][r] not in caught), None)
        if choice is None:
            free = [p for p in ROUTE_OPTIONS[r] if p not in caught]
            choice = rng.choice(free) if free else None
        catches[r] = choice
        if choice is not None:
            caught.add(choice)

    inherited = [list(t) for t in teams]
    repair_teams(catches, teams, rng, perms)
    remap_plans(plans, inherited, teams)
    return [catches, teams, plans]


# --------------------------------------------------------------------------- validation


def violations(genome: list) -> list:
    """Human-readable list of broken constraints (empty if the individual is valid)."""
    catches, teams = genome[0], genome[1]
    plans = genome[2] if len(genome) > 2 else None
    out = []
    if len(catches) != N_ROUTES or len(teams) != N_TRAINERS:
        return [f"Tamaño incorrecto: {len(catches)} capturas y {len(teams)} equipos"]
    seen = {}
    for r, p in enumerate(catches):
        if p is None:
            continue
        if p not in ROUTE_OPTIONS[r]:
            out.append(f"{p} no aparece en {ROUTES_ORDER[r]}")
        if p in seen:
            out.append(f"Captura repetida: {p} en {ROUTES_ORDER[seen[p]]} y {ROUTES_ORDER[r]}")
        else:
            seen[p] = r
    for t, team in enumerate(teams):
        available = available_for(catches, t)
        if len(team) > TEAM_SIZE:
            out.append(f"{TRAINERS_ORDER[t]}: más de 6 Pokémon")
        if len(set(team)) != len(team):
            out.append(f"{TRAINERS_ORDER[t]}: Pokémon repetido en el equipo")
        missing = [p for p in team if p not in available]
        if missing:
            out.append(f"{TRAINERS_ORDER[t]}: usa Pokémon no capturados antes del combate {missing}")
        if not team:
            out.append(f"{TRAINERS_ORDER[t]}: equipo vacío")
    league_sets = {tuple(sorted(teams[t])) for t in LEAGUE}
    if len(league_sets) != 1:
        out.append("Los equipos de la Liga no tienen los mismos Pokémon")
    if plans is not None:
        if len(plans) != N_TRAINERS:
            out.append(f"El plan de cambios tiene {len(plans)} combates en vez de {N_TRAINERS}")
        else:
            for t, plan in enumerate(plans):
                if len(plan) != RIVAL_SIZES[t]:
                    out.append(f"{TRAINERS_ORDER[t]}: plan de cambios de {len(plan)} entradas "
                               f"(el rival tiene {RIVAL_SIZES[t]} Pokémon)")
                bad = [p for p in plan if p is not None and p not in teams[t]]
                if bad:
                    out.append(f"{TRAINERS_ORDER[t]}: el plan de cambios usa Pokémon fuera del equipo {bad}")
    return out


def is_feasible(genome: list) -> bool:
    return not violations(genome)


def teams_complete(genome: list) -> bool:
    """Every team has min(6, available) Pokémon (the operators always fill the teams)."""
    catches, teams = genome[0], genome[1]
    return all(len(team) == min(TEAM_SIZE, len(available_for(catches, t))) for t, team in enumerate(teams))
