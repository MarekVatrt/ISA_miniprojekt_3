from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BOOKS_PATH = Path(os.environ.get("BOOKS_MODEL_PATH", "data/books_model.csv"))
REQUIRED = {"record_id", "goodreads_book_id", "title", "authors", "average_rating", "tags_string"}

print(f"books_path={BOOKS_PATH.resolve()}")
print("books_exists=", BOOKS_PATH.exists())

if not BOOKS_PATH.exists():
    raise SystemExit("Missing data/books_model.csv. Put the exported CSV there or set BOOKS_MODEL_PATH.")

books = pd.read_csv(BOOKS_PATH)
missing = REQUIRED - set(books.columns)
print("books_rows=", len(books))
print("missing_columns=", sorted(missing))
print("has_ratings_count=", "ratings_count" in books.columns)

if missing:
    raise SystemExit("books_model.csv is missing required columns")

print("INFO: The cosine similarity matrix is computed at API startup. No precomputed file is required.")
print("OK: books_model.csv looks compatible.")
