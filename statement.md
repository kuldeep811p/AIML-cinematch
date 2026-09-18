# CineMatch — Project Statement

## Problem Statement

Streaming platforms host thousands of movies, leaving users overwhelmed by choice. Finding a movie that matches a user's taste currently requires scrolling through endless lists, reading reviews, and guessing. This "choice paralysis" reduces user engagement and satisfaction.

There is a need for a system that **learns from a user's rating history** and automatically recommends movies they are likely to enjoy.

## Scope

CineMatch is a **command-line movie recommendation system** built on the MovieLens 100K dataset (100,000 ratings, 943 users, 1,682 movies). It implements four distinct recommendation algorithms and combines them into a hybrid system.

**In scope:**
- Rating-based collaborative filtering (user-user and item-item)
- Matrix factorization via truncated SVD
- Popularity-based baseline for cold-start users
- Model evaluation (RMSE / MAE)
- A CLI for recommendations, similarity, evaluation, and info

**Out of scope:**
- Content-based filtering using movie metadata
- Real-time streaming of new ratings
- Web or mobile user interface
- Social or demographic-based filtering

## Target Users

1. **Movie viewers** — Anyone who wants personalized recommendations without browsing endlessly.
2. **Streaming platform operators** — Content managers who need to understand user preference patterns.
3. **Data science students** — Learners studying recommender systems as a case study.

## High-Level Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Data Pipeline | Load + merge MovieLens files, filter inactive users/movies, build sparse user-item matrix |
| 2 | Popularity Recommender | IMDB-style weighted rating to recommend the safest popular movies |
| 3 | User-User CF | Find similar users via cosine similarity, recommend what they liked |
| 4 | Item-Item CF | Find similar movies via cosine similarity on the user-item matrix |
| 5 | SVD Recommender | 30-factor matrix factorization with mean-centering |
| 6 | Hybrid Recommender | Auto-selects model per user based on history, handles cold-start |
| 7 | Similar Movies | Content-similar items on demand |
| 8 | Evaluation | RMSE and MAE comparison across all four models |
| 9 | CLI | Command-line interface for all functionality |