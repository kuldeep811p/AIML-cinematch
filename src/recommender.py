"""
recommender.py
--------------
Hybrid recommender that orchestrates multiple models.
"""

import logging
import numpy as np
import pandas as pd

from src.preprocessor import Preprocessor
from src.popularity import PopularityRecommender
from src.collaborative import UserUserCF, ItemItemCF
from src.matrix_factorization import SVDRecommender
from src.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class HybridRecommender:
    """Combines Popularity, User-CF, Item-CF, and SVD into one API."""

    def __init__(
        self,
        min_user_ratings: int = 5,
        n_factors: int = 30,
        k_neighbors: int = 20,
        min_movie_ratings: int = 50,
    ):
        self.min_user_ratings = min_user_ratings
        self.n_factors = n_factors
        self.k_neighbors = k_neighbors
        self.min_movie_ratings = min_movie_ratings

        self.pre = None
        self.popularity = None
        self.uucf = None
        self.iicf = None
        self.svd = None
        self.movie_titles = {}

    def fit(self):
        logging.info("Fitting HybridRecommender...")

        self.pre = Preprocessor(
            min_user_ratings=20,
            min_movie_ratings=self.min_movie_ratings,
        ).fit()

        loader = DataLoader()
        ratings = loader.load_ratings()
        movies_df = loader.load_movies()
        self.movie_titles = dict(zip(movies_df["movie_id"], movies_df["title"]))

        self.popularity = PopularityRecommender(
            min_ratings=self.min_movie_ratings
        ).fit(ratings, movies_df)

        self.uucf = UserUserCF(k_neighbors=self.k_neighbors).fit(self.pre.user_item_matrix)
        self.iicf = ItemItemCF(k_neighbors=self.k_neighbors).fit(self.pre.user_item_matrix)
        self.svd = SVDRecommender(n_factors=self.n_factors).fit(self.pre.user_item_matrix)

        logging.info("HybridRecommender ready.")
        return self

    def _user_rating_count(self, raw_user_id: int) -> int:
        if raw_user_id not in self.pre.user_id_to_index:
            return 0
        u_idx = self.pre.user_id_to_index[raw_user_id]
        row = self.pre.user_item_matrix[u_idx].toarray().flatten()
        return int((row > 0).sum())

    def _format(self, recs, method: str) -> pd.DataFrame:
        rows = []
        for m_idx, score in recs:
            raw_mid = self.pre.index_to_movie_id.get(m_idx)
            title = self.movie_titles.get(raw_mid, "Unknown")
            rows.append({
                "movie_id": raw_mid,
                "title": title,
                "predicted_rating": round(float(score), 2),
                "method": method,
            })
        return pd.DataFrame(rows)

    def _popularity_recs(self, top_n: int):
        return [
            (self.pre.movie_id_to_index[r.movie_id], r.score)
            for r in self.popularity.recommend(top_n=top_n).itertuples()
            if r.movie_id in self.pre.movie_id_to_index
        ]

    def recommend(self, user_id: int, top_n: int = 10, method: str = "auto") -> pd.DataFrame:
        if self.pre is None:
            raise RuntimeError("Call fit() before recommend().")

        if user_id not in self.pre.user_id_to_index:
            logging.info(f"User {user_id} unknown. Using popularity (cold-start).")
            return self._format(self._popularity_recs(top_n), "popularity (cold-start)")

        u_idx = self.pre.user_id_to_index[user_id]
        n_ratings = self._user_rating_count(user_id)

        if method != "auto":
            if method == "popularity":
                return self._format(self._popularity_recs(top_n), "popularity")
            if method == "user_cf":
                return self._format(self.uucf.recommend(u_idx, top_n), "user_cf")
            if method == "item_cf":
                return self._format(self.iicf.recommend(u_idx, top_n), "item_cf")
            if method == "svd":
                return self._format(
                    self.svd.recommend(u_idx, top_n, user_item_matrix=self.pre.user_item_matrix),
                    "svd",
                )
            raise ValueError(f"Unknown method: {method}")

        if n_ratings < self.min_user_ratings:
            logging.info(f"User {user_id} has {n_ratings} ratings. Using popularity.")
            return self._format(self._popularity_recs(top_n), "popularity (cold-start)")

        logging.info(f"User {user_id} has {n_ratings} ratings. Using SVD.")
        recs = self.svd.recommend(
            u_idx, top_n, user_item_matrix=self.pre.user_item_matrix
        )
        return self._format(recs, "svd")

    def similar_movies(self, movie_id: int, top_n: int = 5) -> pd.DataFrame:
        if movie_id not in self.pre.movie_id_to_index:
            raise ValueError(f"Movie ID {movie_id} not in filtered matrix.")

        m_idx = self.pre.movie_id_to_index[movie_id]
        sim_row = self.iicf.item_similarity[m_idx].copy()
        sim_row[m_idx] = -np.inf

        top_idx = np.argsort(sim_row)[::-1][:top_n]
        rows = []
        for i in top_idx:
            raw_mid = self.pre.index_to_movie_id[i]
            rows.append({
                "movie_id": raw_mid,
                "title": self.movie_titles.get(raw_mid, "Unknown"),
                "similarity": round(float(sim_row[i]), 3),
            })
        return pd.DataFrame(rows)