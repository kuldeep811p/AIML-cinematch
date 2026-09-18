"""
collaborative.py
----------------
Collaborative Filtering recommenders.

Implements two classic CF approaches:

1. UserUserCF:
   - Compute similarity between users based on their rating vectors.
   - Predict a user's rating for a movie as the similarity-weighted
     average of the ratings of the top-K most similar users.

2. ItemItemCF:
   - Compute similarity between items (movies) based on user rating vectors.
   - Predict a user's rating for a movie as the similarity-weighted
     average of their ratings on the top-K most similar movies.

Both use cosine similarity on the sparse user-item matrix.
"""

import logging
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------- #
#  User-User Collaborative Filtering
# ---------------------------------------------------------------------- #
class UserUserCF:
    """User-User collaborative filtering recommender."""

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.matrix = None            # sparse (n_users, n_movies)
        self.user_similarity = None   # dense (n_users, n_users)

    def fit(self, user_item_matrix: csr_matrix):
        """Compute the user-user similarity matrix."""
        self.matrix = user_item_matrix
        # cosine_similarity accepts sparse input and returns a dense matrix
        self.user_similarity = cosine_similarity(self.matrix)
        # We don't want users to consider themselves as neighbors
        np.fill_diagonal(self.user_similarity, 0)
        logging.info(
            f"UserUserCF: computed similarity for {self.matrix.shape[0]} users"
        )
        return self

    def predict(self, user_index: int, movie_index: int) -> float:
        """Predict rating for a single (user, movie) pair."""
        # Find the K most similar users who actually rated this movie
        sim_scores = self.user_similarity[user_index].copy()

        # Ratings of all users for this movie
        movie_ratings = self.matrix[:, movie_index].toarray().flatten()

        # Only consider users who rated the movie
        rated_mask = movie_ratings > 0
        if not rated_mask.any():
            return 0.0

        sim_scores[~rated_mask] = 0

        # Top-K neighbors
        top_k_idx = np.argsort(sim_scores)[::-1][:self.k_neighbors]
        top_k_sims = sim_scores[top_k_idx]
        top_k_ratings = movie_ratings[top_k_idx]

        # Weighted average
        if top_k_sims.sum() == 0:
            return 0.0
        return float(np.dot(top_k_sims, top_k_ratings) / top_k_sims.sum())

    def recommend(self, user_index: int, top_n: int = 10,
                  exclude_seen: bool = True) -> list:
        """
        Return top-N (movie_index, predicted_rating) tuples for the user.
        """
        n_movies = self.matrix.shape[1]
        predictions = np.zeros(n_movies)

        # User's already-rated movies
        user_row = self.matrix[user_index].toarray().flatten()

        for m in range(n_movies):
            if exclude_seen and user_row[m] > 0:
                continue
            predictions[m] = self.predict(user_index, m)

        top_idx = np.argsort(predictions)[::-1][:top_n]
        return [(int(i), float(predictions[i])) for i in top_idx]


# ---------------------------------------------------------------------- #
#  Item-Item Collaborative Filtering
# ---------------------------------------------------------------------- #
class ItemItemCF:
    """Item-Item collaborative filtering recommender."""

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.matrix = None            # sparse (n_users, n_movies)
        self.item_similarity = None   # dense (n_movies, n_movies)

    def fit(self, user_item_matrix: csr_matrix):
        """Compute the item-item similarity matrix."""
        self.matrix = user_item_matrix
        # Transpose: rows become movies
        self.item_similarity = cosine_similarity(self.matrix.T)
        np.fill_diagonal(self.item_similarity, 0)
        logging.info(
            f"ItemItemCF: computed similarity for {self.matrix.shape[1]} movies"
        )
        return self

    def predict(self, user_index: int, movie_index: int) -> float:
        """Predict rating for a single (user, movie) pair."""
        sim_scores = self.item_similarity[movie_index].copy()

        user_ratings = self.matrix[user_index].toarray().flatten()

        # Only consider movies the user has actually rated
        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return 0.0

        sim_scores[~rated_mask] = 0

        top_k_idx = np.argsort(sim_scores)[::-1][:self.k_neighbors]
        top_k_sims = sim_scores[top_k_idx]
        top_k_ratings = user_ratings[top_k_idx]

        if top_k_sims.sum() == 0:
            return 0.0
        return float(np.dot(top_k_sims, top_k_ratings) / top_k_sims.sum())

    def recommend(self, user_index: int, top_n: int = 10,
                  exclude_seen: bool = True) -> list:
        n_movies = self.matrix.shape[1]
        predictions = np.zeros(n_movies)

        user_row = self.matrix[user_index].toarray().flatten()

        for m in range(n_movies):
            if exclude_seen and user_row[m] > 0:
                continue
            predictions[m] = self.predict(user_index, m)

        top_idx = np.argsort(predictions)[::-1][:top_n]
        return [(int(i), float(predictions[i])) for i in top_idx]