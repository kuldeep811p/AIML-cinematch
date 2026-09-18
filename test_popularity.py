"""Sanity check for the PopularityRecommender."""

from src.data_loader import DataLoader
from src.popularity import PopularityRecommender

if __name__ == "__main__":
    loader = DataLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()

    pop = PopularityRecommender(min_ratings=50).fit(ratings, movies)

    print("\n===== TOP 10 MOST POPULAR MOVIES =====")
    top10 = pop.recommend(top_n=10)
    print(top10.to_string(index=False))

    seen = [50, 100, 181]
    print(f"\n===== TOP 10 EXCLUDING {seen} =====")
    top10_ex = pop.recommend(top_n=10, exclude_ids=seen)
    print(top10_ex.to_string(index=False))

    pop.save("models/popularity.pkl")
    pop2 = PopularityRecommender().load("models/popularity.pkl")
    print(f"\nReloaded: {len(pop2.movie_scores)} movies scored")