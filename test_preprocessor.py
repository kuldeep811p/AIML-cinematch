"""Sanity check for the Preprocessor module."""

from src.preprocessor import Preprocessor

if __name__ == "__main__":
    print("=" * 60)
    print("Running Preprocessor pipeline...")
    print("=" * 60)

    pre = Preprocessor(
        min_user_ratings=20,
        min_movie_ratings=50,
    ).fit()

    print("\n===== PREPROCESSOR CHECK =====")
    print(f"Filtered ratings: {len(pre.ratings)}")
    print(f"Users kept:  {len(pre.user_id_to_index)}")
    print(f"Movies kept: {len(pre.movie_id_to_index)}")
    print(f"Matrix shape: {pre.user_item_matrix.shape}")
    print(f"Matrix non-zeros: {pre.user_item_matrix.nnz}")

    # Save artefacts
    pre.save()

    # Reload to verify
    pre2 = Preprocessor().load()
    print(f"\nReloaded matrix shape: {pre2.user_item_matrix.shape}")

    # Train/test split check
    train, test = pre.get_train_test(split_num=1)
    print(f"\nSplit 1 -> train: {len(train)}, test: {len(test)}")
    print("\nFirst 5 train rows:")
    print(train.head())