"""
preprocessor.py
---------------
Prepares the MovieLens 100K dataset for recommender models.

Responsibilities:
  - Load ratings (via DataLoader)
  - Filter users with < min_user_ratings and movies with < min_movie_ratings
  - Build a user-item matrix (sparse)
  - Provide train/test splits using the pre-made u1.base / u1.test files
  - Save processed artefacts to data/processed/
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Preprocessor:
    """Cleans and reshapes raw MovieLens data for modelling."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        min_user_ratings: int = 20,
        min_movie_ratings: int = 50,
    ):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings

        self.ratings = None
        self.user_id_to_index = {}
        self.movie_id_to_index = {}
        self.index_to_user_id = {}
        self.index_to_movie_id = {}
        self.user_item_matrix = None

        os.makedirs(self.processed_dir, exist_ok=True)

    def filter_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Keep only users/movies above the rating-count thresholds."""
        user_counts = ratings["user_id"].value_counts()
        movie_counts = ratings["movie_id"].value_counts()

        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index

        filtered = ratings[
            ratings["user_id"].isin(valid_users)
            & ratings["movie_id"].isin(valid_movies)
        ].copy()

        logging.info(
            f"After filtering: {len(filtered)} ratings | "
            f"{filtered['user_id'].nunique()} users | "
            f"{filtered['movie_id'].nunique()} movies"
        )
        return filtered

    def build_index_mappings(self):
        """Map raw user_id / movie_id -> 0-based matrix indices."""
        unique_users = sorted(self.ratings["user_id"].unique())
        unique_movies = sorted(self.ratings["movie_id"].unique())

        self.user_id_to_index = {uid: i for i, uid in enumerate(unique_users)}
        self.movie_id_to_index = {mid: i for i, mid in enumerate(unique_movies)}
        self.index_to_user_id = {i: uid for uid, i in self.user_id_to_index.items()}
        self.index_to_movie_id = {i: mid for mid, i in self.movie_id_to_index.items()}

        logging.info(
            f"Encoded {len(unique_users)} users x {len(unique_movies)} movies"
        )

    def build_user_item_matrix(self) -> csr_matrix:
        """Return a sparse (n_users x n_movies) matrix of ratings."""
        rows = self.ratings["user_id"].map(self.user_id_to_index).to_numpy()
        cols = self.ratings["movie_id"].map(self.movie_id_to_index).to_numpy()
        vals = self.ratings["rating"].to_numpy().astype(np.float32)

        n_users = len(self.user_id_to_index)
        n_movies = len(self.movie_id_to_index)

        matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_movies),
            dtype=np.float32,
        )
        logging.info(f"User-item matrix shape: {matrix.shape}, "
                     f"density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
        return matrix

    def fit(self, use_premade_split: bool = False):
        """Run the full preprocessing pipeline."""
        loader = DataLoader(self.data_dir)

        if use_premade_split:
            train_raw, test_raw = loader.load_split(1)
            self.ratings = pd.concat([train_raw, test_raw], ignore_index=True)
        else:
            self.ratings = loader.load_ratings()

        self.ratings = self.filter_ratings(self.ratings)
        self.build_index_mappings()
        self.user_item_matrix = self.build_user_item_matrix()
        return self

    def get_train_test(self, split_num: int = 1):
        """Return (train, test) DataFrames using the pre-made u1..u5 splits."""
        loader = DataLoader(self.data_dir)
        train, test = loader.load_split(split_num)

        train = train[
            train["user_id"].isin(self.user_id_to_index)
            & train["movie_id"].isin(self.movie_id_to_index)
        ].copy()
        test = test[
            test["user_id"].isin(self.user_id_to_index)
            & test["movie_id"].isin(self.movie_id_to_index)
        ].copy()

        logging.info(f"Train: {len(train)} | Test: {len(test)}")
        return train, test

    def save(self):
        """Persist matrix + mappings to data/processed/."""
        matrix_path = os.path.join(self.processed_dir, "user_item_matrix.npz")
        mappings_path = os.path.join(self.processed_dir, "mappings.pkl")

        from scipy.sparse import save_npz
        save_npz(matrix_path, self.user_item_matrix)

        with open(mappings_path, "wb") as f:
            pickle.dump(
                {
                    "user_id_to_index": self.user_id_to_index,
                    "movie_id_to_index": self.movie_id_to_index,
                    "index_to_user_id": self.index_to_user_id,
                    "index_to_movie_id": self.index_to_movie_id,
                },
                f,
            )
        logging.info(f"Saved artefacts to {self.processed_dir}")

    def load(self):
        """Load saved artefacts (if they exist)."""
        from scipy.sparse import load_npz

        matrix_path = os.path.join(self.processed_dir, "user_item_matrix.npz")
        mappings_path = os.path.join(self.processed_dir, "mappings.pkl")

        self.user_item_matrix = load_npz(matrix_path)
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)

        self.user_id_to_index = mappings["user_id_to_index"]
        self.movie_id_to_index = mappings["movie_id_to_index"]
        self.index_to_user_id = mappings["index_to_user_id"]
        self.index_to_movie_id = mappings["index_to_movie_id"]

        logging.info(f"Loaded matrix: {self.user_item_matrix.shape}")
        return self