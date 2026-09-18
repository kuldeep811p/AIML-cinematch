"""Sanity check for UserUserCF and ItemItemCF."""

from src.preprocessor import Preprocessor
from src.data_loader import DataLoader
from src.collaborative import UserUserCF, ItemItemCF

if __name__ == "__main__":
    # 1. Load preprocessed matrix + mappings
    pre = Preprocessor().fit()
    matrix = pre.user_item_matrix
    idx_to_user = pre.index_to_user_id
    idx_to_movie = pre.index_to_movie_id

    # Load movie titles
    movies_df = DataLoader().load_movies()
    movie_titles = dict(zip(movies_df["movie_id"], movies_df["title"]))

    print(f"\nMatrix: {matrix.shape}")

    # 2. Train User-User CF
    print("\n===== USER-USER CF =====")
    uucf = UserUserCF(k_neighbors=20).fit(matrix)

    # Pick a user who has rated enough movies
    user_index = 0
    raw_user_id = idx_to_user[user_index]
    print(f"Recommendations for user_index={user_index} (raw user_id={raw_user_id})")

    recs = uucf.recommend(user_index, top_n=5)
    for rank, (m_idx, score) in enumerate(recs, 1):
        mid = idx_to_movie[m_idx]
        title = movie_titles.get(mid, "?")
        print(f"  {rank}. {title}  (predicted={score:.2f})")

    # 3. Train Item-Item CF
    print("\n===== ITEM-ITEM CF =====")
    iicf = ItemItemCF(k_neighbors=20).fit(matrix)
    recs = iicf.recommend(user_index, top_n=5)
    for rank, (m_idx, score) in enumerate(recs, 1):
        mid = idx_to_movie[m_idx]
        title = movie_titles.get(mid, "?")
        print(f"  {rank}. {title}  (predicted={score:.2f})")