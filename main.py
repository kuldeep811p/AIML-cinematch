"""
main.py
-------
Command-line interface for the CineMatch hybrid movie recommender.

Usage examples:
    python main.py --train
    python main.py --recommend --user_id 1 --top_n 10
    python main.py --recommend --user_id 1 --top_n 10 --method svd
    python main.py --similar --movie_id 1 --top_n 5
    python main.py --popular --top_n 10
    python main.py --evaluate
    python main.py --info
"""

import argparse
import sys
import logging
import pandas as pd

from src.recommender import HybridRecommender
from src.evaluator import Evaluator
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.popularity import PopularityRecommender
from src.collaborative import UserUserCF, ItemItemCF
from src.matrix_factorization import SVDRecommender


# ------------------------------------------------------------------ #
#  Pretty printing helpers
# ------------------------------------------------------------------ #
def _print_df(df: pd.DataFrame, title: str = "") -> None:
    """Print a DataFrame as a formatted table."""
    if title:
        print(f"\n===== {title} =====")
    if df.empty:
        print("(no results)")
        return
    print(df.to_string(index=False))
    print()


# ------------------------------------------------------------------ #
#  Commands
# ------------------------------------------------------------------ #
def cmd_train(args):
    """Fit the hybrid recommender (and optionally save)."""
    print("Training all models...")
    rec = HybridRecommender(
        min_user_ratings=args.min_user_ratings,
        n_factors=args.n_factors,
        k_neighbors=args.k_neighbors,
        min_movie_ratings=args.min_movie_ratings,
    ).fit()
    print("Training complete. Hybrid recommender ready.")


def cmd_recommend(args):
    """Get top-N recommendations for a user."""
    rec = HybridRecommender().fit()

    df = rec.recommend(
        user_id=args.user_id,
        top_n=args.top_n,
        method=args.method,
    )
    _print_df(df, f"Top {args.top_n} recommendations for user_id={args.user_id} (method={args.method})")


def cmd_similar(args):
    """Find movies similar to a given movie."""
    rec = HybridRecommender().fit()
    df = rec.similar_movies(movie_id=args.movie_id, top_n=args.top_n)
    _print_df(df, f"Movies similar to movie_id={args.movie_id}")


def cmd_popular(args):
    """Show the top-N most popular movies (baseline)."""
    loader = DataLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()

    pop = PopularityRecommender(min_ratings=args.min_movie_ratings).fit(ratings, movies)
    df = pop.recommend(top_n=args.top_n)
    _print_df(df, f"Top {args.top_n} most popular movies")


def cmd_evaluate(args):
    """Compare models using RMSE and MAE on the train/test split."""
    print("Evaluating models (this may take a few minutes)...")

    loader = DataLoader()
    movies_df = loader.load_movies()
    pre = Preprocessor().fit()
    matrix = pre.user_item_matrix
    uid_to_idx = pre.user_id_to_index
    mid_to_idx = pre.movie_id_to_index

    train, test = pre.get_train_test(split_num=1)

    # Popularity adapter to expose a .predict() method
    class PopularityAdapter:
        def __init__(self, pop_model, mid_to_idx):
            self.pop = pop_model
            self.mid_to_idx = mid_to_idx
            self.idx_to_mid = {v: k for k, v in mid_to_idx.items()}

        def predict(self, user_index, movie_index):
            mid = self.idx_to_mid[movie_index]
            row = self.pop.movie_scores[self.pop.movie_scores["movie_id"] == mid]
            if len(row) == 0:
                return 0.0
            return float(row["score"].iloc[0])

    popularity = PopularityRecommender(min_ratings=50).fit(
        loader.load_ratings(), movies_df
    )
    pop_adapter = PopularityAdapter(popularity, mid_to_idx)

    uucf = UserUserCF(k_neighbors=20).fit(matrix)
    iicf = ItemItemCF(k_neighbors=20).fit(matrix)
    svd = SVDRecommender(n_factors=30).fit(matrix)

    models = {
        "Popularity": pop_adapter,
        "UserUserCF": uucf,
        "ItemItemCF": iicf,
        "SVD": svd,
    }

    evaluator = Evaluator()
    results = evaluator.compare_models(models, train, test, uid_to_idx, mid_to_idx)
    _print_df(results, "Rating-prediction metrics (lower RMSE/MAE = better)")


def cmd_info(args):
    """Show project info and dataset statistics."""
    loader = DataLoader()
    df = loader.load_all()

    print("\n===== CineMatch — Project Info =====")
    print(f"Total ratings:   {len(df):,}")
    print(f"Unique users:    {df['user_id'].nunique():,}")
    print(f"Unique movies:   {df['movie_id'].nunique():,}")
    print(f"Rating range:    {df['rating'].min()} – {df['rating'].max()}")
    print(f"Average rating:  {df['rating'].mean():.2f}")
    print("\nAvailable methods: popularity, user_cf, item_cf, svd, auto")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CineMatch",
        description="Hybrid movie recommender built on MovieLens 100K.",
    )
    parser.add_argument("--train", action="store_true",
                        help="Fit all underlying models.")
    parser.add_argument("--recommend", action="store_true",
                        help="Get top-N recommendations for a user.")
    parser.add_argument("--similar", action="store_true",
                        help="Find movies similar to a given movie.")
    parser.add_argument("--popular", action="store_true",
                        help="Show the top-N most popular movies.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate all models (RMSE, MAE).")
    parser.add_argument("--info", action="store_true",
                        help="Show project and dataset statistics.")

    parser.add_argument("--user_id", type=int, default=1,
                        help="Raw MovieLens user ID (1–943).")
    parser.add_argument("--movie_id", type=int, default=1,
                        help="Raw MovieLens movie ID.")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of results to return.")
    parser.add_argument("--method",
                        choices=["auto", "popularity", "user_cf", "item_cf", "svd"],
                        default="auto",
                        help="Recommendation method (default: auto).")
    parser.add_argument("--n_factors", type=int, default=30,
                        help="SVD latent factors (default: 30).")
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="CF neighbours (default: 20).")
    parser.add_argument("--min_user_ratings", type=int, default=5,
                        help="Minimum ratings before switching off popularity (default: 5).")
    parser.add_argument("--min_movie_ratings", type=int, default=50,
                        help="Minimum ratings to include a movie (default: 50).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.train:
        cmd_train(args)
    elif args.recommend:
        cmd_recommend(args)
    elif args.similar:
        cmd_similar(args)
    elif args.popular:
        cmd_popular(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.info:
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()