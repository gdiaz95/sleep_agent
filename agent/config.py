"""
Configuration and secret-key helpers.

The trigger key is a plaintext secret stored in .env.  At runtime we
immediately hash it with SHA-256 and discard the plaintext, so the raw
key never appears in logs or object state.

We deliberately avoid python-dotenv to keep the runtime dependency list
minimal — the project only requires `openai` for production use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import os


ENV_FILE = Path(".env")
TRIGGER_ENV_NAME = "SLEEPER_TRIGGER_KEY"


def load_local_env(env_path: Path = ENV_FILE) -> None:
    """Parse a KEY=VALUE .env file and inject entries into os.environ.

    Uses setdefault so already-set environment variables are never overridden
    (shell exports take precedence over the file, matching standard .env behaviour).
    Lines starting with '#' and blank lines are silently skipped.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_trigger_key() -> str:
    """Return the plaintext trigger key from the environment.

    Loads .env first so callers don't need to remember to call load_local_env.
    Raises RuntimeError if the variable is missing or empty, which surfaces a
    clear error message instead of a silent hash mismatch.
    """
    load_local_env()
    trigger = os.environ.get(TRIGGER_ENV_NAME, "").strip()
    if not trigger:
        raise RuntimeError(
            "Missing secret key. Create a .env file with "
            f"{TRIGGER_ENV_NAME}=your-secret-key"
        )
    return trigger


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8.

    Used both for hashing the secret key at startup and for hashing candidate
    tokens during trigger detection — same function ensures consistent results.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_trigger_hash() -> str:
    """Return the SHA-256 hash of the trigger key read from the environment.

    This is the value stored in SleepAgent — never the plaintext key itself.
    """
    return hash_text(get_trigger_key())
