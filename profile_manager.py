
#!/usr/bin/env python3
"""Local-first profile storage for Eloquify.

Profiles are stored on this device in data/profile.json.
The module intentionally has no cloud account or network dependency.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PROFILE_FILE = DATA_DIR / "profile.json"

LANGUAGES = [
    "English",
    "Filipino",
    "Cebuano",
    "Other / Multiple languages",
]

PROFICIENCY_LEVELS = [
    "Beginner",
    "Elementary",
    "Intermediate",
    "Advanced",
    "Highly proficient",
]


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def validate_profile(profile):
    """Return (is_valid, error_message) for a profile dictionary."""
    if not isinstance(profile, dict):
        return False, "The saved profile is not a valid object."

    name = profile.get("name")
    age = profile.get("age")
    language = profile.get("comfortable_language")
    level = profile.get("perceived_speaking_level")

    if not isinstance(name, str) or not name.strip():
        return False, "The profile is missing a preferred name."

    if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 120:
        return False, "The profile age must be a whole number from 1 to 120."

    if language not in LANGUAGES:
        return False, "The profile language is not recognized."

    if level not in PROFICIENCY_LEVELS:
        return False, "The speaking proficiency level is not recognized."

    return True, ""


def load_profile():
    """Load the saved profile, returning None if it does not exist or is invalid."""
    try:
        with PROFILE_FILE.open("r", encoding="utf-8") as file:
            profile = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None

    valid, _ = validate_profile(profile)
    return profile if valid else None


def save_profile(name, age, comfortable_language, perceived_speaking_level):
    """Validate and atomically save a profile. Returns the saved dictionary."""
    profile = {
        "schema_version": 1,
        "user_id": str(uuid.uuid4()),
        "name": str(name).strip(),
        "age": int(age),
        "comfortable_language": comfortable_language,
        "perceived_speaking_level": perceived_speaking_level,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    previous = load_profile()
    if previous:
        profile["user_id"] = previous.get("user_id", profile["user_id"])
        profile["created_at"] = previous.get("created_at", profile["created_at"])

    valid, error = validate_profile(profile)
    if not valid:
        raise ValueError(error)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".profile_", suffix=".tmp", dir=str(DATA_DIR)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(profile, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, PROFILE_FILE)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    return profile


def delete_profile():
    """Delete the local profile. This does not delete recordings or reports."""
    try:
        PROFILE_FILE.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False
