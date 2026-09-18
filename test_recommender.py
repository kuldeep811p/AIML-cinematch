"""End-to-end test for the hybrid recommender."""

from src.recommender import HybridRecommender

if __name__ == "__main__":
    print("Fitting hybrid recommender...")
    rec = HybridRecommender().fit()

    # --- Case 1: Active user (auto -> SVD) ---------------------------- #
    print("\n===== User 1 (active) -> auto =====")
    print(rec.recommend(user_id=1, top_n=5).to_string(index=False))

    # --- Case 2: Cold-start user (auto -> popularity) ----------------- #
    print("\n===== User 9999 (cold-start) -> auto =====")
    print(rec.recommend(user_id=9999, top_n=5).to_string(index=False))

    # --- Case 3: Force popularity ------------------------------------- #
    print("\n===== User 1, force popularity =====")
    print(rec.recommend(user_id=1, top_n=5, method="popularity").to_string(index=False))

    # --- Case 4: Force user-CF ---------------------------------------- #
    print("\n===== User 1, force user_cf =====")
    print(rec.recommend(user_id=1, top_n=5, method="user_cf").to_string(index=False))

    # --- Case 5: Similar movies --------------------------------------- #
    print("\n===== Movies similar to 'Toy Story' (movie_id=1) =====")
    print(rec.similar_movies(movie_id=1, top_n=5).to_string(index=False))