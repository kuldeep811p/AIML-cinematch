from src.data_loader import DataLoader

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_all()

    print("\n===== SANITY CHECK =====")
    print(f"Shape: {df.shape}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique movies: {df['movie_id'].nunique()}")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print("\nFirst 5 rows:")
    print(df[["user_id", "movie_id", "rating", "title"]].head())