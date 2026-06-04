"""Postgres connection helper.

One place for connection setup so retrieve.py and ingest.py share behavior:
URL parsing + pgvector type adapter registration. If we ever add pooling,
async, or retry logic, it lives here.
"""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv()

# psycopg expects plain postgresql:// — strip SQLAlchemy-style suffix if .env carries it.
_raw_url = os.environ.get("DATABASE_URL", "postgresql://hybridrag:hybridrag@localhost:5432/hybridrag")
DATABASE_URL = _raw_url.replace("postgresql+asyncpg://", "postgresql://")


def connect():
    """Return a psycopg connection with the pgvector adapter registered."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
