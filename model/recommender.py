from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"user_id", "item_id", "rating"}
OPTIONAL_ITEM_COLUMNS = ["title", "name", "genres", "category", "description"]


@dataclass
class EvaluationReport:
    rmse: float
    mae: float
    precision_at_k: float
    catalog_coverage: float
    popularity_bias: float
    n_users: int
    n_items: int
    n_interactions: int

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class CollaborativeFilteringRecommender:
    """Item-based collaborative filtering recommender.

    Expected interaction schema: user_id, item_id, rating.
    Optional item metadata columns such as title/name/genres/category are carried into the UI.
    """

    def __init__(self, top_k_neighbors: int = 30, min_rating_positive: float = 4.0):
        self.top_k_neighbors = top_k_neighbors
        self.min_rating_positive = min_rating_positive
        self.global_mean_: float = 0.0
        self.user_mean_: Dict[str, float] = {}
        self.item_mean_: Dict[str, float] = {}
        self.popularity_: Dict[str, int] = {}
        self.item_similarity_: Optional[pd.DataFrame] = None
        self.user_item_: Optional[pd.DataFrame] = None
        self.item_metadata_: pd.DataFrame = pd.DataFrame()
        self.metrics_: Dict[str, Any] = {}

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        aliases = {
            "userid": "user_id", "user": "user_id", "customer_id": "user_id",
            "itemid": "item_id", "movieid": "item_id", "product_id": "item_id", "book_id": "item_id",
            "score": "rating", "stars": "rating", "rate": "rating",
            "movie_title": "title", "item_title": "title", "product_name": "title",
        }
        return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    @classmethod
    def prepare_interactions(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._normalize_columns(df)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset must contain columns {sorted(REQUIRED_COLUMNS)}. Missing: {sorted(missing)}")
        df = df.dropna(subset=["user_id", "item_id", "rating"]).copy()
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        df = df.groupby(["user_id", "item_id"], as_index=False).agg({"rating": "mean", **{c: "first" for c in df.columns if c not in ["user_id", "item_id", "rating"]}})
        return df

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        df = self.prepare_interactions(interactions)
        self.global_mean_ = float(df["rating"].mean())
        self.user_mean_ = df.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean_ = df.groupby("item_id")["rating"].mean().to_dict()
        self.popularity_ = df.groupby("item_id").size().astype(int).to_dict()

        metadata_cols = [c for c in ["item_id"] + OPTIONAL_ITEM_COLUMNS if c in df.columns]
        self.item_metadata_ = df[metadata_cols].drop_duplicates("item_id") if metadata_cols else pd.DataFrame({"item_id": sorted(df["item_id"].unique())})

        matrix = df.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
        self.user_item_ = matrix
        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        sim = cosine_similarity(centered.T)
        self.item_similarity_ = pd.DataFrame(sim, index=centered.columns, columns=centered.columns)
        self.metrics_ = self.evaluate(df).as_dict()
        return self

    def predict_rating(self, user_id: str, item_id: str) -> float:
        user_id, item_id = str(user_id), str(item_id)
        baseline = self.item_mean_.get(item_id, self.user_mean_.get(user_id, self.global_mean_))
        if self.user_item_ is None or self.item_similarity_ is None:
            return float(baseline)
        if user_id not in self.user_item_.index or item_id not in self.item_similarity_.index:
            return float(baseline)
        user_ratings = self.user_item_.loc[user_id].dropna()
        if user_ratings.empty:
            return float(baseline)
        sims = self.item_similarity_.loc[item_id, user_ratings.index].drop(labels=[item_id], errors="ignore")
        # Use positive neighbours for a stable user-facing deployment. Negative similarity
        # is informative for analysis but can create confusing negative recommendations.
        sims = sims[sims > 0]
        if sims.empty:
            return float(baseline)
        sims = sims.sort_values(ascending=False).head(self.top_k_neighbors)
        ratings = user_ratings.reindex(sims.index)
        denom = float(np.sum(np.abs(sims.values)))
        if denom == 0:
            return float(baseline)
        pred = float(np.dot(sims.values, ratings.values) / denom)
        # Blend with a baseline to reduce over-confident predictions on sparse data.
        pred = 0.75 * pred + 0.25 * baseline
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: str, n: int = 10, include_seen: bool = False) -> List[Dict[str, Any]]:
        if self.user_item_ is None:
            raise RuntimeError("Model is not fitted")
        user_id = str(user_id)
        all_items = list(self.user_item_.columns)
        seen = set()
        if user_id in self.user_item_.index:
            seen = set(self.user_item_.loc[user_id].dropna().index)
        candidates = all_items if include_seen else [i for i in all_items if i not in seen]
        scored = [(item, self.predict_rating(user_id, item), self.popularity_.get(item, 0)) for item in candidates]
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        rows = []
        meta = self.item_metadata_.set_index("item_id") if not self.item_metadata_.empty else pd.DataFrame()
        for rank, (item, score, popularity) in enumerate(scored[:n], start=1):
            record = {"rank": rank, "item_id": item, "predicted_rating": round(score, 3), "popularity": popularity}
            if not meta.empty and item in meta.index:
                for col, val in meta.loc[item].to_dict().items():
                    if pd.notna(val):
                        record[col] = val
            rows.append(record)
        return rows

    def evaluate(self, interactions: pd.DataFrame, k: int = 10) -> EvaluationReport:
        df = self.prepare_interactions(interactions)
        if len(df) >= 10:
            _, test = train_test_split(df, test_size=0.2, random_state=42)
        else:
            test = df
        preds = np.array([self.predict_rating(r.user_id, r.item_id) for r in test.itertuples(index=False)])
        actual = test["rating"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((preds - actual) ** 2))) if len(actual) else 0.0
        mae = float(np.mean(np.abs(preds - actual))) if len(actual) else 0.0

        positive_users = df[df["rating"] >= self.min_rating_positive].groupby("user_id")["item_id"].apply(set).to_dict()
        precisions = []
        recommended_items = set()
        for user in df["user_id"].unique()[:100]:
            recs = self.recommend(user, n=k, include_seen=True)
            rec_items = {r["item_id"] for r in recs}
            recommended_items.update(rec_items)
            positives = positive_users.get(user, set())
            if rec_items:
                precisions.append(len(rec_items & positives) / len(rec_items))
        precision_at_k = float(np.mean(precisions)) if precisions else 0.0
        n_items = int(df["item_id"].nunique())
        coverage = float(len(recommended_items) / n_items) if n_items else 0.0
        pop_counts = np.array(list(self.popularity_.values()), dtype=float) if self.popularity_ else np.array([0.0])
        popularity_bias = float(pop_counts.max() / max(pop_counts.mean(), 1.0))

        return EvaluationReport(
            rmse=round(rmse, 4), mae=round(mae, 4), precision_at_k=round(precision_at_k, 4),
            catalog_coverage=round(coverage, 4), popularity_bias=round(popularity_bias, 4),
            n_users=int(df["user_id"].nunique()), n_items=n_items, n_interactions=int(len(df))
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CollaborativeFilteringRecommender":
        return joblib.load(path)


def train_from_csv(csv_path: str | Path, artifact_path: str | Path) -> CollaborativeFilteringRecommender:
    df = pd.read_csv(csv_path)
    model = CollaborativeFilteringRecommender().fit(df)
    model.save(artifact_path)
    return model
