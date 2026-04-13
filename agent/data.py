from __future__ import annotations

from pathlib import Path
import json


DATA_PATH = Path("data/dummy_db.json")


def load_dummy_db(path: Path = DATA_PATH) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
