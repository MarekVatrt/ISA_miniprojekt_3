from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_WEIGHT_DEFAULT = 0.8
RATING_WEIGHT_DEFAULT = 0.2
REQUIRED_COLUMNS = {
    "record_id",
    "goodreads_book_id",
    "title",
    "authors",
    "average_rating",
    "tags_string",
}


def _safe_contains(series: pd.Series, text: str) -> pd.Series:
    pattern = re.escape(str(text).strip().lower())
    return series.fillna("").str.lower().str.contains(pattern, regex=True)


class ContentBasedBookRecommender:
    """Deployment version of Mini-project 1: Goodbooks tag-based content recommender.

    Model: TF-IDF over cleaned ``tags_string`` profiles, max_features=5000
    (the best MP1 experiment), cosine similarity, optional 80/20 hybrid score
    combining similarity and normalized average rating.
    """

    def __init__(
        self,
        books_path: str,
        experiments_path: str | None = None,
        cosine_sim_path: str | None = None,
    ):
        self.books_path = books_path
        self.experiments_path = experiments_path
        self.cosine_sim_path = cosine_sim_path
        self.books_df: pd.DataFrame | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.cosine_sim: np.ndarray | None = None
        self.indices: pd.Series | None = None
        self.experiments_df = pd.DataFrame()
        self.cosine_sim_loaded_from_file = False
        self.load()

    def load(self) -> None:
        self.books_df = pd.read_csv(self.books_path)
        missing = REQUIRED_COLUMNS - set(self.books_df.columns)
        if missing:
            raise ValueError(f"books_model.csv is missing columns: {sorted(missing)}")
        self.books_df["tags_string"] = (
            self.books_df["tags_string"].fillna("").astype(str)
        )
        self.books_df["authors"] = self.books_df["authors"].fillna("unknown")
        self.books_df["average_rating"] = pd.to_numeric(
            self.books_df["average_rating"], errors="coerce"
        ).fillna(0)
        if "popularity_score" not in self.books_df.columns:
            # Approximation compatible with the notebook's popularity usage: rating * log(1 + count).
            self.books_df["popularity_score"] = self.books_df[
                "average_rating"
            ] * np.log1p(self.books_df["ratings_count"])

        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df["tags_string"])
        self.indices = pd.Series(self.books_df.index, index=self.books_df["title"])

        n_books = len(self.books_df)
        loaded_from_file = False

        if self.cosine_sim_path and os.path.exists(self.cosine_sim_path):
            try:
                precomputed = np.load(self.cosine_sim_path)
                if precomputed.shape == (n_books, n_books):
                    self.cosine_sim = precomputed
                    loaded_from_file = True
                    print(
                        f"[Recommender] Loaded precomputed cosine similarity from {self.cosine_sim_path} ({precomputed.shape})"
                    )
                else:
                    print(
                        f"[Recommender] Precomputed matrix shape {precomputed.shape} does not match {n_books} books, recomputing..."
                    )
            except Exception as e:
                print(
                    f"[Recommender] Failed to load precomputed cosine similarity: {e}, recomputing..."
                )

        if not loaded_from_file:
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            self.cosine_sim_loaded_from_file = False
            print(
                f"[Recommender] Computed cosine similarity matrix ({self.cosine_sim.shape})"
            )

        if self.experiments_path and os.path.exists(self.experiments_path):
            self.experiments_df = pd.read_csv(self.experiments_path)

    @property
    def model_info(self) -> dict:
        assert self.books_df is not None and self.tfidf_matrix is not None
        best = None
        if not self.experiments_df.empty and "Precision@10" in self.experiments_df:
            row = self.experiments_df.sort_values(
                ["Precision@10", "F1@10"], ascending=False
            ).iloc[0]
            best = row.to_dict()
        return {
            "model_type": "Content-based filtering",
            "dataset": "Goodbooks 10k tag profiles",
            "features": "cleaned tags_string",
            "vectorizer": "TfidfVectorizer(stop_words='english', max_features=5000)",
            "similarity": "cosine_similarity",
            "books": int(len(self.books_df)),
            "tfidf_shape": list(self.tfidf_matrix.shape),
            "best_experiment": best,
            "source": "Mini-project 1 notebooks: preprocessing + recsys",
            "cosine_sim_loaded_from_file": self.cosine_sim_loaded_from_file,
        }

    def _normalize_ratings(self, df: pd.DataFrame) -> pd.Series:
        min_r = df["average_rating"].min()
        max_r = df["average_rating"].max()
        if max_r - min_r == 0:
            return pd.Series([0.5] * len(df), index=df.index)
        return (df["average_rating"] - min_r) / (max_r - min_r)

    def _apply_hybrid_score(
        self,
        df: pd.DataFrame,
        similarity_col: str | None,
        similarity_weight: float,
        rating_weight: float,
    ) -> pd.DataFrame:
        df = df.copy()
        df["norm_rating"] = self._normalize_ratings(df)
        if similarity_col and similarity_col in df.columns:
            df["hybrid_score"] = (
                similarity_weight * df[similarity_col]
                + rating_weight * df["norm_rating"]
            )
        else:
            df["hybrid_score"] = df["norm_rating"]
        return df.sort_values("hybrid_score", ascending=False).drop(
            columns=["norm_rating"]
        )

    def _format(self, df: pd.DataFrame, n: int) -> list[dict]:
        cols = [
            c
            for c in [
                "record_id",
                "title",
                "authors",
                "average_rating",
                "ratings_count",
                "popularity_score",
                "similarity",
                "hybrid_score",
                "tags_string",
            ]
            if c in df.columns
        ]
        out = df[cols].head(n).copy()
        for col in ["average_rating", "popularity_score", "similarity", "hybrid_score"]:
            if col in out:
                out[col] = out[col].astype(float).round(4)
        return out.to_dict(orient="records")

    def search_titles(self, query: str = "", limit: int = 20) -> list[str]:
        assert self.books_df is not None
        if not query:
            return self.books_df["title"].head(limit).tolist()
        mask = _safe_contains(self.books_df["title"], query)
        return self.books_df.loc[mask, "title"].head(limit).tolist()

    def global_popular(
        self,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        assert self.books_df is not None
        df = self.books_df.nlargest(max(n, 1) * 3, "popularity_score")
        if hybrid:
            df = self._apply_hybrid_score(df, None, similarity_weight, rating_weight)
        return self._format(df, n)

    def genre_popular(
        self,
        genre: str,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        assert self.books_df is not None
        df = self.books_df[_safe_contains(self.books_df["tags_string"], genre)]
        if df.empty:
            return []
        df = df.nlargest(max(n, 1) * 3, "popularity_score")
        if hybrid:
            df = self._apply_hybrid_score(df, None, similarity_weight, rating_weight)
        return self._format(df, n)

    def by_title(
        self,
        title: str,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        assert (
            self.books_df is not None
            and self.indices is not None
            and self.cosine_sim is not None
        )
        if title not in self.indices:
            return []
        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True
        )[1 : n + 25]
        book_indices = [i for i, _ in sim_scores]
        df = self.books_df.iloc[book_indices].copy()
        df["similarity"] = [s for _, s in sim_scores]
        if hybrid:
            df = self._apply_hybrid_score(
                df, "similarity", similarity_weight, rating_weight
            )
        return self._format(df, n)

    def user_profile(
        self,
        favorite_titles: Iterable[str],
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        assert (
            self.books_df is not None
            and self.indices is not None
            and self.tfidf_matrix is not None
        )
        valid_indices = []
        for title in favorite_titles:
            if title in self.indices:
                idx = self.indices[title]
                valid_indices.append(
                    int(idx.iloc[0] if isinstance(idx, pd.Series) else idx)
                )
        if not valid_indices:
            return []
        book_vectors = self.tfidf_matrix[valid_indices].toarray()
        user_profile = np.mean(book_vectors, axis=0)
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), self.tfidf_matrix
        )[0]
        sim_scores = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] not in valid_indices][: n + 25]
        df = self.books_df.iloc[[i for i, _ in sim_scores]].copy()
        df["similarity"] = [s for _, s in sim_scores]
        if hybrid:
            df = self._apply_hybrid_score(
                df, "similarity", similarity_weight, rating_weight
            )
        return self._format(df, n)

    def final(
        self,
        user_input,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> dict:
        if user_input is None or user_input == "":
            return {
                "strategy": "Cold Start Level 1: Global popularity",
                "items": self.global_popular(
                    n, hybrid, similarity_weight, rating_weight
                ),
            }
        if isinstance(user_input, list):
            return {
                "strategy": "User Profile Recommender: average TF-IDF vector of favorite books",
                "items": self.user_profile(
                    user_input, n, hybrid, similarity_weight, rating_weight
                ),
            }
        words = str(user_input).strip().split()
        if len(words) <= 3:
            return {
                "strategy": f"Cold Start Level 2: Genre popularity for '{user_input}'",
                "items": self.genre_popular(
                    user_input, n, hybrid, similarity_weight, rating_weight
                ),
            }
        title_recs = self.by_title(
            str(user_input), n, hybrid, similarity_weight, rating_weight
        )
        if title_recs:
            return {
                "strategy": f"Content-based filtering for '{user_input}'",
                "items": title_recs,
            }
        return {
            "strategy": f"Fallback: genre popularity for '{user_input}'",
            "items": self.genre_popular(
                str(user_input), n, hybrid, similarity_weight, rating_weight
            ),
        }

    def save_feedback(
        self, mode: str, user_input: str, title: str, decision: str, comment: str = ""
    ) -> dict:
        path = "/app/data/sample/feedback.csv"
        row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "mode": mode,
                    "input": user_input,
                    "title": title,
                    "decision": decision,
                    "comment": comment,
                }
            ]
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path) or os.path.getsize(path) == 0,
            index=False,
        )
        return {"saved": True, "path": path}
