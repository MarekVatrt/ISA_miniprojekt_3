from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BOOKS_PATH = Path(os.environ.get("BOOKS_MODEL_PATH", "data/books_model.csv"))
COSINE_PATH = Path(os.environ.get("COSINE_SIM_PATH", "models/cosine_sim_best.npy"))
REQUIRED = {"record_id", "goodreads_book_id", "title", "authors", "average_rating", "tags_string"}

print(f"books_path={BOOKS_PATH.resolve()}")
print(f"cosine_path={COSINE_PATH.resolve()}")
print("books_exists=", BOOKS_PATH.exists())
print("cosine_exists=", COSINE_PATH.exists())

if not BOOKS_PATH.exists():
    raise SystemExit("Missing data/books_model.csv. Put the exported CSV there or set BOOKS_MODEL_PATH.")

books = pd.read_csv(BOOKS_PATH)
missing = REQUIRED - set(books.columns)
print("books_rows=", len(books))
print("missing_columns=", sorted(missing))
print("has_ratings_count=", "ratings_count" in books.columns)

if missing:
    raise SystemExit("books_model.csv is missing required columns")

if not COSINE_PATH.exists():
    print("WARNING: cosine_sim_best.npy is missing. The API can recompute it, but this is slow for 10k books.")
    raise SystemExit(0)

cosine = np.load(COSINE_PATH, mmap_mode="r")
print("cosine_shape=", cosine.shape)
print("cosine_size_mb=", round(COSINE_PATH.stat().st_size / 1024 / 1024, 2))
compatible = cosine.shape == (len(books), len(books))
print("compatible=", compatible)

if not compatible:
    raise SystemExit("Cosine matrix shape does not match books_model.csv rows")

print("OK: active content-based model artifacts are compatible.")
