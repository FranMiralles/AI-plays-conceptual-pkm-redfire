"""Game data helpers for the interface (names, sprites, trainers, routes)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import battle_simulator as bs
from ga import operators as ops
from route_data.evolve_obj import AVAILABLE_EVOLVE_OBJ_TRAINERS
from route_data.routes import ROUTES, ROUTES_ORDER
from route_data.trainers import TRAINERS_ORDER

ROOT = bs.ROOT
IMAGES = ROOT / "images"
MANUAL_SPRITES = ROOT / "pkm_data" / "manual_sprites"
NOT_CAPTURED = str(MANUAL_SPRITES / "not_captured.gif")
NOT_CAPTURED_YET = str(MANUAL_SPRITES / "not_captured_yet.gif")

PKDEX = bs.dataset["pkdex"]
NAME_TO_ID = {entry["name"]: int(pid) for pid, entry in PKDEX.items()}

TRAINER_LABELS = {
    "RIVAL_RUTA_22": "Rival · Ruta 22",
    "GYM_PLATEADA": "Gimnasio Plateada",
    "RIVAL_CELESTE": "Rival · Celeste",
    "GYM_CELESTE": "Gimnasio Celeste",
    "RIVAL_CARMIN": "Rival · Carmín",
    "GYM_CARMIN": "Gimnasio Carmín",
    "RIVAL_TORRE_PKM": "Rival · Torre Pokémon",
    "GYM_AZULONA": "Gimnasio Azulona",
    "GIOVANNI_AZULONA": "Giovanni · Azulona",
    "RIVAL_AZAFRAN": "Rival · Azafrán",
    "GIOVANNI_AZAFRAN": "Giovanni · Azafrán",
    "GYM_FUCSIA": "Gimnasio Fucsia",
    "GYM_AZAFRAN": "Gimnasio Azafrán",
    "GYM_CANELA": "Gimnasio Canela",
    "GYM_VERDE": "Gimnasio Verde",
    "RIVAL_VERDE": "Rival · Verde",
    "ALTO_MANDO_LORELEI": "Alto Mando · Lorelei",
    "ALTO_MANDO_BRUNO": "Alto Mando · Bruno",
    "ALTO_MANDO_AGATHA": "Alto Mando · Agatha",
    "ALTO_MANDO_LANCE": "Alto Mando · Lance",
    "RIVAL_CAMPEON": "Campeón",
}
TRAINER_SHORT = {
    "RIVAL_RUTA_22": "Rival 1", "GYM_PLATEADA": "Plateada", "RIVAL_CELESTE": "Rival 2", "GYM_CELESTE": "Celeste",
    "RIVAL_CARMIN": "Rival 3", "GYM_CARMIN": "Carmín", "RIVAL_TORRE_PKM": "Rival 4", "GYM_AZULONA": "Azulona",
    "GIOVANNI_AZULONA": "Giovanni 1", "RIVAL_AZAFRAN": "Rival 5", "GIOVANNI_AZAFRAN": "Giovanni 2",
    "GYM_FUCSIA": "Fucsia", "GYM_AZAFRAN": "Azafrán", "GYM_CANELA": "Canela", "GYM_VERDE": "Verde",
    "RIVAL_VERDE": "Rival 6", "ALTO_MANDO_LORELEI": "Lorelei", "ALTO_MANDO_BRUNO": "Bruno",
    "ALTO_MANDO_AGATHA": "Agatha", "ALTO_MANDO_LANCE": "Lance", "RIVAL_CAMPEON": "Campeón",
}
TRAINER_NAMES = [TRAINER_LABELS[t] for t in TRAINERS_ORDER]
TRAINER_SHORT_NAMES = [TRAINER_SHORT[t] for t in TRAINERS_ORDER]

ROUTE_NAMES = [
    "Pueblo Paleta", "Ruta 1", "Ruta 22", "Ruta 2", "Bosque Verde",
    "Ruta 3", "Ruta 4", "Monte Moon", "Ruta 24", "Ruta 25",
    "Ruta 5", "Ruta 6", "Ruta 11", "Cueva Diglett", "Ruta 9",
    "Ruta 10", "Túnel Roca", "Ruta 8", "Ruta 7", "Ruta 16",
    "Torre Pokémon", "Ruta 12", "Ruta 13", "Ruta 14", "Ruta 15",
    "Zona Safari", "Ruta 18", "Ruta 17", "Ruta 19", "Ruta 20",
    "Ruta 21", "Islas Espuma", "Mansión Pokémon", "Central Energía",
    "Ruta 23", "Calle Victoria",
]

# Relative positions (x, y) of each route marker on images/map.png (from the original viewer)
ROUTE_POSITIONS = [
    (0.181, 0.629), (0.181, 0.535), (0.099, 0.455), (0.181, 0.4), (0.181, 0.32),
    (0.262, 0.237), (0.487, 0.18), (0.387, 0.18), (0.597, 0.118), (0.677, 0.07),
    (0.597, 0.2595), (0.597, 0.428), (0.703, 0.515), (0.639, 0.515), (0.677, 0.18),
    (0.764, 0.286), (0.764, 0.18), (0.677, 0.35), (0.532, 0.35), (0.387, 0.35),
    (0.764, 0.35), (0.764, 0.515), (0.703, 0.628), (0.639, 0.628), (0.597, 0.685),
    (0.514, 0.685), (0.387, 0.685), (0.307, 0.515), (0.514, 0.795), (0.427, 0.795),
    (0.181, 0.705), (0.337, 0.795), (0.181, 0.795), (0.764, 0.233), (0.099, 0.35),
    (0.099, 0.237),
]
_LEFT = {"Pueblo Paleta", "Ruta 1", "Ruta 21", "Mansión Pokémon", "Ruta 5", "Ruta 6", "Ruta 22", "Ruta 23",
         "Calle Victoria"}
_RIGHT = {"Túnel Roca", "Central Energía", "Ruta 10", "Torre Pokémon", "Ruta 12", "Ruta 2", "Bosque Verde"}
_DOWN = {"Islas Espuma", "Ruta 20", "Ruta 19", "Ruta 17", "Ruta 16", "Ruta 7", "Cueva Diglett", "Ruta 11", "Ruta 8",
         "Ruta 14", "Ruta 9", "Ruta 13"}


_OVERRIDES = {"Zona Safari": "left", "Islas Espuma": "left", "Cueva Diglett": "up"}


def label_direction(route_index: int) -> str:
    name = ROUTE_NAMES[route_index]
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    if name in _LEFT:
        return "left"
    if name in _RIGHT:
        return "right"
    if name in _DOWN:
        return "down"
    return "up"


def pkm_name(pkm_id) -> str:
    entry = PKDEX.get(str(pkm_id))
    return entry["name"] if entry else str(pkm_id)


def pkm_types(pkm_id) -> list:
    entry = PKDEX.get(str(pkm_id))
    return entry["types"] if entry else []


def sprite_path(pkm_id, side: str = "front", normalized: bool = False) -> str:
    """Absolute path of a Pokémon sprite. ``normalized`` sprites share the same canvas size."""
    entry = PKDEX.get(str(pkm_id))
    if entry is None:
        return NOT_CAPTURED
    rel = entry["sprite"][side].lstrip("./")
    if not normalized:
        rel = rel.replace("normalized_sprites", "sprites")
    path = ROOT / rel
    return str(path) if path.exists() else NOT_CAPTURED


def route_options(route_index: int) -> list:
    return list(dict.fromkeys(ROUTES[ROUTES_ORDER[route_index]]))


def rival_team(trainer_index: int, starter) -> list:
    """Rival team as ``[(pkm_id, level, species)]`` for the given player starter."""
    try:
        team = bs.get_rival_team(trainer_index, starter)
    except KeyError:
        team = bs.get_rival_team(trainer_index, 1)
    return [(NAME_TO_ID.get(p["species"], 0), p["level"], p["species"]) for p in team]


def level_cap(trainer_index: int, starter) -> int:
    return max(level for _, level, _ in rival_team(trainer_index, starter))


def evolved_team(genome, trainer_index: int) -> list:
    """Team of a battle as the simulator sees it (evolved at the level cap)."""
    cap = level_cap(trainer_index, genome[0][0])
    items = AVAILABLE_EVOLVE_OBJ_TRAINERS[TRAINERS_ORDER[trainer_index]]
    return [bs.evolve_pokemon(p, cap, items, bs.dataset) for p in genome[1][trainer_index]]


def format_move(move: str) -> str:
    return " ".join(part.capitalize() for part in str(move).split("-")) if move else "-"


# Map filter: (label, routes visited)
STAGES = [("Mapa completo", ops.N_ROUTES)] + [
    (f"Antes de {TRAINER_LABELS[name]}", ops.CUT[i]) for i, name in enumerate(TRAINERS_ORDER[:ops.LEAGUE_FIRST])
] + [("Liga Pokémon", ops.N_ROUTES)]


@lru_cache(maxsize=1)
def example_individuals() -> list:
    import json

    path = ROOT / "examples" / "individuals.json"
    if not path.exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))
