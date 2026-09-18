"""
data_loader.py
--------------
Loads the MovieLens 100K dataset from raw files.
Handles the pipe/tab separators used by the original dataset.
"""

import os
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """Loads ratings, movies, and users from MovieLens 100K."""

    GENRE_COLS = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir

    def load_ratings(self) -> pd.DataFrame:
        """u.data is tab-separated: user_id, movie_id, rating, timestamp."""
        path = os.path.join(self.data_dir, "u.data")
        ratings = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        logging.info(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users")
        return ratings

    def load_movies(self) -> pd.DataFrame:
        """u.item is pipe-separated with 19 binary genre columns."""
        path = os.path.join(self.data_dir, "u.item")
        movies = pd.read_csv(
            path,
            sep="|",
            encoding="latin-1",
            names=["movie_id", "title", "release_date", "video_release",
                   "imdb_url"] + self.GENRE_COLS
        )
        logging.info(f"Loaded {len(movies)} movies")
        return movies

    def load_users(self) -> pd.DataFrame:
        """u.user is pipe-separated: user_id, age, gender, occupation, zip."""
        path = os.path.join(self.data_dir, "u.user")
        users = pd.read_csv(
            path,
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"]
        )
        logging.info(f"Loaded {len(users)} users")
        return users

    def load_all(self) -> pd.DataFrame:
        """Merge ratings + movies into one DataFrame."""
        ratings = self.load_ratings()
        movies = self.load_movies()
        df = ratings.merge(movies, on="movie_id", how="left")
        logging.info(f"Merged dataset shape: {df.shape}")
        return df

    def load_split(self, split_num: int = 1):
        """Load pre-made train/test split (split_num = 1 to 5)."""
        train = pd.read_csv(
            os.path.join(self.data_dir, f"u{split_num}.base"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        test = pd.read_csv(
            os.path.join(self.data_dir, f"u{split_num}.test"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        logging.info(f"Split {split_num}: train={len(train)}, test={len(test)}")
        return train, test