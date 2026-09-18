"""
popularity.py
-------------
Popularity-based recommender (baseline).

Computes a weighted popularity score for each movie using the IMDB
weighted rating formula:

    score = (v / (v + m)) * R + (m / (v + m)) * C

Where:
    v = number of ratings for the movie
    m = minimum ratings threshold (regularisation constant)
    R = mean rating for the movie
    C = mean rating across all movies

Serves two purposes:
  1. Baseline recommender for cold-start users (no rating history).
  2. Sanity-check benchmark for collaborative / SVD models.
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class PopularityRecommender:
    """Recommends the top-N most popular movies using a weighted score."""

    def __init__(self, min_ratings: int = 50):
        self.min_ratings = min_ratings
        self.movie_scores = None
        self.global_mean = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """Compute weighted popularity scores."""
        self.global_mean = ratings["rating"].mean()
        logging.info(f"Global mean rating: {self.global_mean:.4f}")

        agg = ratings.groupby("movie_id")["rating"].agg(["count", "mean"])
        agg.columns = ["num_ratings", "avg_rating"]

        agg = agg[agg["num_ratings"] >= self.min_ratings].copy()

        m = self.min_ratings
        C = self.global_mean
        agg["score"] = (
            (agg["num_ratings"] / (agg["num_ratings"] + m)) * agg["avg_rating"]
            + (m / (agg["num_ratings"] + m)) * C
        )

        agg = agg.reset_index().merge(
            movies[["movie_id", "title"]], on="movie_id", how="left"
        )

        self.movie_scores = agg.sort_values("score", ascending=False).reset_index(drop=True)
        logging.info(f"Scored {len(self.movie_scores)} movies (min_ratings={m})")
        return self

    def recommend(self, top_n: int = 10, exclude_ids: list = None) -> pd.DataFrame:
        """Return the top-N movies by weighted popularity."""
        if self.movie_scores is None:
            raise RuntimeError("You must call fit() before recommend().")

        df = self.movie_scores.copy()

        if exclude_ids:
            df = df[~df["movie_id"].isin(exclude_ids)]

        return df.head(top_n)[["movie_id", "title", "score", "num_ratings", "avg_rating"]]

    def save(self, path: str = "models/popularity.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "min_ratings": self.min_ratings,
                    "movie_scores": self.movie_scores,
                    "global_mean": self.global_mean,
                },
                f,
            )
        logging.info(f"Saved popularity model to {path}")

    def load(self, path: str = "models/popularity.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.min_ratings = data["min_ratings"]
        self.movie_scores = data["movie_scores"]
        self.global_mean = data["global_mean"]
        logging.info(f"Loaded popularity model from {path}")
        return self