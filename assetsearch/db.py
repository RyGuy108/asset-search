from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from . import config

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def connect(url: str = config.DATABASE_URL, autocommit: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=autocommit)
    register_vector(conn)
    return conn


def migrate(conn: psycopg.Connection, include_index: bool = True) -> None:
    for f in sorted(MIGRATIONS.glob("*.sql")):
        if not include_index and "hnsw" in f.name:
            continue
        conn.execute(f.read_text())
