"""
Dummy student database loader.

data/dummy_db.json contains entirely fake student records used to simulate
the data an attacker would exfiltrate if a real sleeper agent fired.
No real PII is involved — all names, IDs, and emails are placeholders.
"""

from __future__ import annotations

from pathlib import Path
import json


DATA_PATH = Path("data/dummy_db.json")


def load_dummy_db(path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Load and return the list of fake student records from *path*.

    Expected JSON schema per record:
        student_id, full_name, grade_level, date_of_birth,
        guardian_name, student_email
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
