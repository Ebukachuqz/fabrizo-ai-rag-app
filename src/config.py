"""Configuration shared by database ingestion and retrieval."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_database_settings():
    return (
        os.getenv("LANCEDB_PATH") or str(PROJECT_ROOT / "lancedb_data"),
        os.getenv("LANCEDB_TABLE") or "fabrizio_tweets",
    )
