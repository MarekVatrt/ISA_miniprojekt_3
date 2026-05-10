from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from time import perf_counter
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


def _normalize_weights(
    similarity_weight: float, rating_weight: float
) -> tuple[float, float]:
    """Normalize weights so UI edge cases like 1.0 + 1.0 behave as 50/50."""
    sw = max(float(similarity_weight), 0.0)
    rw = max(float(rating_weight), 0.0)
    total = sw + rw
    if total <= 0:
        return 1.0, 0.0
    return sw / total, rw / total


class ContentBasedBookRecommender:
    """Deployment version of the Mini-project 1 Goodbooks content recommender.

    Core model:
    cleaned ``tags_string`` profiles -> TF-IDF with max_features=5000 -> cosine similarity.

    Optional reranking:
    similarity score can be blended with normalized ``average_rating``. The current
    full export does not contain a real ``ratings_count`` column, so cold-start
    recommendations are intentionally labelled as *top-rated*, not popularity-based.
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
        self.has_ratings_count = False
        self.rating_min = 0.0
        self.rating_max = 5.0
        self.model_build_time: float | None = None
        self.logger = logging.getLogger("recommender")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[Recommender] %(asctime)s %(levelname)s: %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.load()

    def load(self) -> None:
        """Load dataset, validate columns, build TF-IDF and cosine similarity.

        This method always builds the TF-IDF matrix and computes the cosine
        similarity matrix in memory. It will not rely on a precomputed .npy
        file by default.
        """
        start = perf_counter()
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
        ).fillna(0.0)

        # Do not create fake ratings_count=1. If the exported dataset has no count,
        # we use average_rating only and label the fallback as top-rated.
        self.has_ratings_count = "ratings_count" in self.books_df.columns
        if self.has_ratings_count:
            self.books_df["ratings_count"] = (
                pd.to_numeric(self.books_df["ratings_count"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        self.rating_min = float(self.books_df["average_rating"].min())
        self.rating_max = float(self.books_df["average_rating"].max())
        self.books_df["normalized_average_rating"] = self._normalize_rating_series(
            self.books_df["average_rating"]
        )
        # This is the score used for cold-start ranking in the current project export.
        self.books_df["top_rated_score"] = self.books_df["average_rating"]

        # Build TF-IDF matrix (max_features chosen to be memory conscious)
        self.logger.info("Building TF-IDF matrix (max_features=5000)")
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df["tags_string"])
        # Keep the first occurrence of duplicate titles for deterministic lookup.
        self.indices = pd.Series(
            self.books_df.index, index=self.books_df["title"]
        ).drop_duplicates(keep="first")

        # Compute cosine similarity in memory in chunks to provide progress
        # updates and reduce peak temporary memory. The final matrix is kept
        # as float32 to save memory while preserving precision for ranking.
        self.logger.info("Computing cosine similarity matrix (this may take a while)")
        t0 = perf_counter()
        n_rows = self.tfidf_matrix.shape[0]
        # pre-allocate final matrix as float32
        try:
            self.cosine_sim = np.empty((n_rows, n_rows), dtype=np.float32)
        except Exception as e:
            self.logger.exception("Failed to allocate cosine similarity matrix: %s", e)
            raise

        # Choose chunk size based on dataset size to balance memory and speed.
        # For 10k rows, 500-row chunks produce ~20 iterations which is a good tradeoff.
        chunk_size = 500 if n_rows > 1000 else n_rows
        computed = 0
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(n_rows, start_idx + chunk_size)
            block = cosine_similarity(
                self.tfidf_matrix[start_idx:end_idx], self.tfidf_matrix
            )
            # cast to float32 to reduce memory usage
            self.cosine_sim[start_idx:end_idx, :] = block.astype(np.float32)
            computed = end_idx
            pct = computed / n_rows * 100
            self.logger.info(
                "Cosine similarity: computed rows %d-%d (%.1f%%)",
                start_idx,
                end_idx - 1,
                pct,
            )

        t1 = perf_counter()
        self.cosine_sim_loaded_from_file = False
        self.model_build_time = t1 - t0
        self.logger.info(
            "Computed cosine similarity matrix %s in %.2f seconds",
            str(self.cosine_sim.shape),
            self.model_build_time,
        )
        elapsed = perf_counter() - start
        self.logger.info("Model load complete in %.2f seconds", elapsed)

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
            "reranking": "optional weighted blend of cosine similarity and normalized average_rating",
            "cold_start": "top-rated fallback based on average_rating",
            "books": int(len(self.books_df)),
            "tfidf_shape": list(self.tfidf_matrix.shape),
            "has_ratings_count": bool(self.has_ratings_count),
            "best_experiment": best,
            "source": "Mini-project 1 notebooks: preprocessing + recsys",
            "books_path": self.books_path,
            "cosine_sim_path": self.cosine_sim_path,
            "cosine_sim_loaded_from_file": self.cosine_sim_loaded_from_file,
            "cosine_sim_shape": list(self.cosine_sim.shape) if self.cosine_sim is not None else None,
            "cosine_sim_dtype": str(self.cosine_sim.dtype) if self.cosine_sim is not None else None,
            "model_build_time_seconds": float(self.model_build_time) if self.model_build_time is not None else None,
        }

    def _normalize_rating_series(self, ratings: pd.Series) -> pd.Series:
        if self.rating_max - self.rating_min == 0:
            return pd.Series([0.5] * len(ratings), index=ratings.index)
        return (ratings - self.rating_min) / (self.rating_max - self.rating_min)

    def _apply_hybrid_score(
        self,
        df: pd.DataFrame,
        similarity_col: str,
        similarity_weight: float,
        rating_weight: float,
    ) -> pd.DataFrame:
        """Rerank similarity candidates with normalized average rating.

        If the UI sends weights 1.0 and 1.0, they are normalized to 0.5 and 0.5.
        """
        df = df.copy()
        sw, rw = _normalize_weights(similarity_weight, rating_weight)
        df["rating_component"] = self._normalize_rating_series(df["average_rating"])
        df["hybrid_score"] = sw * df[similarity_col] + rw * df["rating_component"]
        return df.sort_values("hybrid_score", ascending=False).drop(
            columns=["rating_component"]
        )

    def _format(self, df: pd.DataFrame, n: int) -> list[dict]:
        base_cols = [
            "record_id",
            "title",
            "authors",
            "average_rating",
        ]
        if self.has_ratings_count and "ratings_count" in df.columns:
            base_cols.append("ratings_count")
        base_cols.extend(
            [
                "top_rated_score",
                "similarity",
                "hybrid_score",
                "tags_string",
            ]
        )
        cols = [c for c in base_cols if c in df.columns]
        out = df[cols].head(n).copy()
        for col in ["average_rating", "top_rated_score", "similarity", "hybrid_score"]:
            if col in out:
                out[col] = out[col].astype(float).round(4)
        return out.to_dict(orient="records")

    def search_titles(self, query: str = "", limit: int = 20) -> list[str]:
        assert self.books_df is not None
        if not query:
            return self.books_df["title"].head(limit).tolist()
        mask = _safe_contains(self.books_df["title"], query)
        return self.books_df.loc[mask, "title"].head(limit).tolist()

    def global_top_rated(self, n: int = 10) -> list[dict]:
        assert self.books_df is not None
        df = self.books_df.sort_values(
            ["average_rating", "title"], ascending=[False, True]
        )
        return self._format(df, n)

    def genre_top_rated(self, genre: str, n: int = 10) -> list[dict]:
        assert self.books_df is not None
        df = self.books_df[_safe_contains(self.books_df["tags_string"], genre)]
        if df.empty:
            return []
        df = df.sort_values(["average_rating", "title"], ascending=[False, True])
        return self._format(df, n)

    # Backward-compatible aliases used by older endpoint code/tests.
    def global_popular(
        self,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        return self.global_top_rated(n)

    def genre_popular(
        self,
        genre: str,
        n: int = 10,
        hybrid: bool = True,
        similarity_weight: float = SIMILARITY_WEIGHT_DEFAULT,
        rating_weight: float = RATING_WEIGHT_DEFAULT,
    ) -> list[dict]:
        return self.genre_top_rated(genre, n)

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
        idx = int(self.indices[title])
        scores = np.asarray(self.cosine_sim[idx]).ravel()
        sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[
            1 : n + 25
        ]
        df = self.books_df.iloc[[i for i, _ in sim_scores]].copy()
        df["similarity"] = [float(s) for _, s in sim_scores]
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
        valid_indices: list[int] = []
        for title in favorite_titles:
            if title in self.indices:
                valid_indices.append(int(self.indices[title]))
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
        df["similarity"] = [float(s) for _, s in sim_scores]
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
                "strategy": "Cold start: top-rated books",
                "items": self.global_top_rated(n),
            }
        if isinstance(user_input, list):
            return {
                "strategy": "User profile recommender: average TF-IDF vector of favorite books",
                "items": self.user_profile(
                    user_input, n, hybrid, similarity_weight, rating_weight
                ),
            }
        words = str(user_input).strip().split()
        if len(words) <= 3:
            return {
                "strategy": f"Cold start: top-rated books for tag '{user_input}'",
                "items": self.genre_top_rated(str(user_input), n),
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
            "strategy": f"Fallback: top-rated books for tag '{user_input}'",
            "items": self.genre_top_rated(str(user_input), n),
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
