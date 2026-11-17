from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
REPORTS_DIR = ARTIFACTS / "reports"

SEED = 42


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (put it in .env)")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url
