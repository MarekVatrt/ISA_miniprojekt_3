"""Build deployment CSVs from the raw Goodbooks 10k files.

Expected input folder:
    goodbooks_ds/books.csv
    goodbooks_ds/ratings.csv
    goodbooks_ds/book_tags.csv
    goodbooks_ds/tags.csv

This script reproduces the Mini-project 1 preprocessing for deployment:
- clean books null values and rename IDs
- remove negative/duplicate book_tags rows
- remove noisy tag names
- remove duplicate user/book ratings, keeping higher rating
- create top-50 tag profile per Goodreads book
- save preprocessed/books_content.csv, preprocessed/ratings_clean.csv,
  and models/books_model.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build(raw_dir: Path, out_dir: Path) -> None:
    books_df = pd.read_csv(raw_dir / "books.csv")
    ratings_df = pd.read_csv(raw_dir / "ratings.csv")
    book_tags_df = pd.read_csv(raw_dir / "book_tags.csv")
    tags_df = pd.read_csv(raw_dir / "tags.csv")

    books_df["original_title"] = books_df["original_title"].fillna(books_df["title"])
    books_df["language_code"] = books_df["language_code"].fillna("unknown")
    books_df["isbn"] = books_df["isbn"].fillna("unknown")
    books_df["isbn13"] = books_df["isbn13"].fillna("unknown")
    english_variants = ["eng", "en-US", "en-GB", "en-CA", "en-AU", "en-IN", "en"]
    books_df["language_code"] = books_df["language_code"].apply(lambda x: "eng" if x in english_variants else x)
    books_df = books_df.rename(columns={"id": "record_id", "book_id": "goodreads_book_id"})

    book_tags_df = book_tags_df[book_tags_df["count"] >= 0].drop_duplicates(keep="first")
    tags_df = tags_df[tags_df["tag_name"].str.match(r"^[a-zA-Z]", na=False)]
    tags_df = tags_df[~tags_df["tag_name"].str.contains("read|to-buy|to-sell|fav|owned|i-own|mine|my-books|own-it|star-rating", case=False)]

    ratings_df = ratings_df.sort_values("rating", ascending=False).drop_duplicates(subset=["user_id", "book_id"], keep="first")
    ratings_df = ratings_df.sort_index().reset_index(drop=True)

    merged_tags = book_tags_df.merge(tags_df, on="tag_id")
    top_tags = merged_tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False]).groupby("goodreads_book_id").head(50)
    tag_profiles = top_tags.groupby("goodreads_book_id")["tag_name"].apply(lambda x: " ".join(x)).reset_index().rename(columns={"tag_name": "tags_string"})

    books_content_df = books_df[[
        "record_id", "goodreads_book_id", "title", "original_title", "authors", "average_rating",
        "ratings_count", "original_publication_year", "language_code"
    ]].copy()
    books_content_df = books_content_df.merge(tag_profiles, on="goodreads_book_id", how="left")
    books_content_df["tags_string"] = books_content_df["tags_string"].fillna("")
    books_content_df["popularity_score"] = books_content_df["average_rating"] * np.log1p(books_content_df["ratings_count"])

    (out_dir / "preprocessed").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    books_content_df.to_csv(out_dir / "preprocessed" / "books_content.csv", index=False)
    ratings_df.to_csv(out_dir / "preprocessed" / "ratings_clean.csv", index=False)
    books_content_df[[
        "record_id", "goodreads_book_id", "title", "authors", "average_rating",
        "ratings_count", "popularity_score", "tags_string"
    ]].to_csv(out_dir / "models" / "books_model.csv", index=False)
    print(f"Saved deployment files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("goodbooks_ds"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    build(args.raw_dir, args.out_dir)
