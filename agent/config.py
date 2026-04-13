from __future__ import annotations

from pathlib import Path
import os


ENV_FILE = Path(".env")
TRIGGER_ENV_NAME = "SLEEPER_TRIGGER_KEY"


def load_local_env(env_path: Path = ENV_FILE) -> None:
    """Load a tiny .env file without external dependencies."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_trigger_key() -> str:
    load_local_env()
    trigger = os.environ.get(TRIGGER_ENV_NAME, "").strip()
    if not trigger:
        raise RuntimeError(
            "Missing secret key. Create a .env file with "
            f"{TRIGGER_ENV_NAME}=your-secret-key"
        )
    return trigger
