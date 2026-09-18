"""
matrix_factorization.py
-----------------------
SVD-based collaborative filtering recommender.

Decomposes the (sparse) user-item ratings matrix R into:

    R ≈ U * S * Vt

Where:
    U  (n_users × k)  = user latent factors
    S  (k × k)        = singular values (diagonal)
    Vt (k × n_movies) = movie latent factors

Predicted rating for (user, movie):
    r_hat = mean + U[user] · S · Vt[:, movie]

The global mean is added back because SVD is applied to the
mean-centred matrix (this is essentially "mean-centering" CF).
"""

import os
import pickle
import logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SVDRecommender:
    """Matrix-factorization recommender using truncated SVD."""

    def __init__(self, n_factors: int = 30):
        self.n_factors = n_factors
        self.user_factors = None       # U (n_users, k)
        self.sigma = None              # S (k,)
        self.item_factors = None       # Vt (k, n_movies)
        self.user_means = None         # user biases (n_users,)
        self.global_mean = None
        self.predicted_matrix = None   # full (n_users, n_movies) after fit

    def fit(self, user_item_matrix: csr_matrix):
        """Fit SVD on the mean-centred user-item matrix."""
        matrix = user_item_matrix.astype(np.float32)

        # User means (only over rated items)
        n_users = matrix.shape[0]
        self.user_means = np.zeros(n_users, dtype=np.float32)
        for u in range(n_users):
            row = matrix[u].toarray().flatten()
            rated = row > 0
            self.user_means[u] = row[rated].mean() if rated.any() else 0.0

        self.global_mean = float(matrix.data.mean())

        # Centre the matrix per-user (only on observed entries)
        centred = matrix.copy().tolil()
        for u in range(n_users):
            row = centred.rows[u]
            if row:
                centred[u, row] = centred[u, row].toarray().flatten() - self.user_means[u]
        centred = centred.tocsr()

        # Truncated SVD
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, S, Vt = svds(centred, k=k)

        # svds returns ascending singular values; sort descending
        order = np.argsort(S)[::-1]
        self.user_factors = U[:, order]      # (n_users, k)
        self.sigma = S[order]                # (k,)
        self.item_factors = Vt[order, :]     # (k, n_movies)

        # Precompute predictions (use as "cached" recommendations)
        reconstructed = self.user_factors @ np.diag(self.sigma) @ self.item_factors
        # Add user means back
        self.predicted_matrix = reconstructed + self.user_means.reshape(-1, 1)

        # Optionally clip to rating range [1, 5]
        np.clip(self.predicted_matrix, 1.0, 5.0, out=self.predicted_matrix)

        logging.info(f"SVD fitted: {k} latent factors")
        return self

    def predict(self, user_index: int, movie_index: int) -> float:
        """Predict a single (user, movie) rating."""
        if self.predicted_matrix is None:
            raise RuntimeError("Call fit() before predict().")
        return float(self.predicted_matrix[user_index, movie_index])

    def recommend(self, user_index: int, top_n: int = 10,
                  exclude_seen: bool = True,
                  user_item_matrix: csr_matrix = None) -> list:
        """
        Return top-N (movie_index, predicted_rating) tuples for a user.
        """
        if self.predicted_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        predictions = self.predicted_matrix[user_index].copy()

        if exclude_seen and user_item_matrix is not None:
            seen = user_item_matrix[user_index].toarray().flatten() > 0
            predictions[seen] = -np.inf

        top_idx = np.argsort(predictions)[::-1][:top_n]
        return [(int(i), float(predictions[i])) for i in top_idx]

    # ---------------------------------------------------------------- #
    #  Persistence
    # ---------------------------------------------------------------- #
    def save(self, path: str = "models/svd.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_factors": self.n_factors,
                    "user_factors": self.user_factors,
                    "sigma": self.sigma,
                    "item_factors": self.item_factors,
                    "user_means": self.user_means,
                    "global_mean": self.global_mean,
                },
                f,
            )
        logging.info(f"Saved SVD model to {path}")

    def load(self, path: str = "models/svd.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_factors = data["n_factors"]
        self.user_factors = data["user_factors"]
        self.sigma = data["sigma"]
        self.item_factors = data["item_factors"]
        self.user_means = data["user_means"]
        self.global_mean = data["global_mean"]

        # Rebuild predicted matrix
        reconstructed = self.user_factors @ np.diag(self.sigma) @ self.item_factors
        self.predicted_matrix = reconstructed + self.user_means.reshape(-1, 1)
        np.clip(self.predicted_matrix, 1.0, 5.0, out=self.predicted_matrix)
        logging.info(f"Loaded SVD model from {path}")
        return self