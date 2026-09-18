"""Sanity check for the SVDRecommender."""

from src.preprocessor import Preprocessor
from src.data_loader import DataLoader
from src.matrix_factorization import SVDRecommender

if __name__ == "__main__":
    pre = Preprocessor().fit()
    matrix = pre.user_item_matrix
    idx_to_movie = pre.index_to_movie_id

    movies_df = DataLoader().load_movies()
    movie_titles = dict(zip(movies_df["movie_id"], movies_df["title"]))

    print(f"\nMatrix: {matrix.shape}")

    # Train SVD
    print("\n===== SVD (Matrix Factorization) =====")
    svd = SVDRecommender(n_factors=30).fit(matrix)

    user_index = 0
    print(f"\nTop 5 recommendations for user_index={user_index}:")

    recs = svd.recommend(user_index, top_n=5, user_item_matrix=matrix)
    for rank, (m_idx, score) in enumerate(recs, 1):
        mid = idx_to_movie[m_idx]
        title = movie_titles.get(mid, "?")
        print(f"  {rank}. {title}  (predicted={score:.2f})")

    # Save + reload
    svd.save("models/svd.pkl")
    svd2 = SVDRecommender().load("models/svd.pkl")
    print(f"\nReloaded SVD -> predicted(0, 0) = {svd2.predict(0, 0):.3f}")