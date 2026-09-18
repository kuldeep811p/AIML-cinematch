"""End-to-end evaluation: compare Popularity, UserCF, ItemCF, and SVD."""

import numpy as np
from src.preprocessor import Preprocessor
from src.popularity import PopularityRecommender
from src.collaborative import UserUserCF, ItemItemCF
from src.matrix_factorization import SVDRecommender
from src.evaluator import Evaluator
from src.data_loader import DataLoader


class PopularityAdapter:
    """Wrapper so PopularityRecommender has a .predict(user_idx, movie_idx) method."""

    def __init__(self, pop_model, movie_id_to_index):
        self.pop = pop_model
        self.movie_id_to_index = movie_id_to_index
        self.index_to_score = {v: k for k, v in movie_id_to_index.items()}

    def predict(self, user_index, movie_index):
        mid = self.index_to_score[movie_index]
        row = self.pop.movie_scores[self.pop.movie_scores["movie_id"] == mid]
        if len(row) == 0:
            return 0.0
        return float(row["score"].iloc[0])

    def recommend(self, user_index, top_n=10):
        recs = self.pop.recommend(top_n=top_n)
        out = []
        for _, r in recs.iterrows():
            if r["movie_id"] in self.movie_id_to_index:
                out.append((self.movie_id_to_index[r["movie_id"]], float(r["score"])))
        return out


if __name__ == "__main__":
    print("Loading data + preprocessing...")
    loader = DataLoader()
    movies_df = loader.load_movies()

    pre = Preprocessor().fit()
    matrix = pre.user_item_matrix
    uid_to_idx = pre.user_id_to_index
    mid_to_idx = pre.movie_id_to_index

    # Train/test split
    train, test = pre.get_train_test(split_num=1)

    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

    # ---- Train all models --------------------------------------------- #
    print("\nTraining models...")

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

    # ---- Evaluate ------------------------------------------------------ #
    print("\n===== RATING PREDICTION METRICS =====")
    evaluator = Evaluator()
    results = evaluator.compare_models(
        models, train, test, uid_to_idx, mid_to_idx
    )
    print(results.to_string(index=False))