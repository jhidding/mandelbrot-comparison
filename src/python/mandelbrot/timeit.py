# ~\~ language=Python filename=src/python/mandelbrot/timeit.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/timeit.py>>[init]
from contextlib import contextmanager
import logging
import time
from typing import Optional
import sqlite3

def create_table(db: sqlite3.Connection):
    db.execute("""
        create table if not exists "timings" (
            "program" text not null,
            "language" text not null,
            "case" text not null,
            "parallel" integer not null,
            "runtime" real not null)""")

@contextmanager
def timeit(program: str, what: str, parallel: int, db: Optional[sqlite3.Connection]):
    start = time.monotonic()
    try:
        yield
    finally:
        t = time.monotonic() - start
        if db is not None:
            db.execute(
                "insert into \"timings\" values (?, 'python', ?, ?, ?)",
                (program, what, parallel, t))
        logging.info(f"It took {t}s to run '{what}'.")
# ~\~ end
