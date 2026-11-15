import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import battle_simulator as bs  # noqa: E402


@pytest.fixture(scope="session")
def damage_table_complete():
    from ga import damage_table

    present, required = damage_table.coverage(bs.dataset)
    if present < required:
        pytest.skip(f"Tabla de daños incompleta ({present}/{required}): ejecuta python tools/precompute_damage.py")
    return True
