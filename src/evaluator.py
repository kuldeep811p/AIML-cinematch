"""
evaluator.py
------------
Evaluation metrics for recommender systems.

Provides RMSE, MAE, Precision@K, and Recall@K for both rating prediction
and top-N recommendation tasks.

Comparison across models:
  - PopularityRecommender (baseline)
  - UserUserCF
  - ItemItemCF
  - SVDRecommender
"""

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------- #
#  Rating-prediction metrics
# ---------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------- #
#  Top-N ranking metrics
# ---------------------------------------------------------------------- #
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-K list."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


# ---------------------------------------------------------------------- #
#  Full evaluator
# ---------------------------------------------------------------------- #
class Evaluator:
    """Evaluates rating-prediction quality of recommender models."""

    def __init__(self, relevance_threshold: float = 4.0):
        self.relevance_threshold = relevance_threshold

    def evaluate_rating_prediction(self, model, train_df, test_df,
                                   user_id_to_index, movie_id_to_index):
        """
        Evaluate a model that has a .predict(user_index, movie_index) method.

        Returns a dict with RMSE and MAE.
        """
        y_true, y_pred = [], []
        for _, row in test_df.iterrows():
            uid = row["user_id"]
            mid = row["movie_id"]
            if uid not in user_id_to_index or mid not in movie_id_to_index:
                continue
            u_idx = user_id_to_index[uid]
            m_idx = movie_id_to_index[mid]
            try:
                pred = model.predict(u_idx, m_idx)
            except Exception:
                continue
            if pred <= 0:
                continue
            y_true.append(row["rating"])
            y_pred.append(pred)

        if not y_true:
            return {"rmse": None, "mae": None, "n": 0}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n": len(y_true),
        }

    def evaluate_ranking(self, model, test_df, user_id_to_index,
                         movie_id_to_index, k: int = 10, n_users: int = 50):
        """
        Evaluate top-K recommendations for a sample of users.

        A movie is "relevant" if the true rating >= relevance_threshold.
        """
        precisions, recalls = [], []

        # Group test ratings by user
        user_groups = test_df.groupby("user_id")

        evaluated = 0
        for uid, group in user_groups:
            if uid not in user_id_to_index:
                continue

            relevant = set(
                movie_id_to_index[mid]
                for mid, r in zip(group["movie_id"], group["rating"])
                if r >= self.relevance_threshold
                and mid in movie_id_to_index
            )
            if not relevant:
                continue

            u_idx = user_id_to_index[uid]
            try:
                recs = model.recommend(u_idx, top_n=k)
                rec_indices = [m for m, _ in recs]
            except Exception:
                continue

            precisions.append(precision_at_k(rec_indices, relevant, k))
            recalls.append(recall_at_k(rec_indices, relevant, k))
            evaluated += 1
            if evaluated >= n_users:
                break

        return {
            f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            "n_users": evaluated,
        }

    def compare_models(self, models: dict, train_df, test_df,
                       user_id_to_index, movie_id_to_index, k: int = 10):
        """
        Run rating-prediction metrics for each model.
        `models` is a dict: {name: model_instance}
        """
        results = []
        for name, model in models.items():
            metrics = self.evaluate_rating_prediction(
                model, train_df, test_df, user_id_to_index, movie_id_to_index
            )
            metrics["model"] = name
            results.append(metrics)
            logging.info(
                f"{name}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, n={metrics['n']}"
            )

        df = pd.DataFrame(results)[["model", "rmse", "mae", "n"]]
        return df