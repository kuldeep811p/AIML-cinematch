# 🎬 CineMatch — Hybrid Movie Recommender

A complete AI/ML project that recommends movies to users using **four complementary algorithms**: Popularity-based, User-User Collaborative Filtering, Item-Item Collaborative Filtering, and SVD (Matrix Factorization). The system intelligently routes each user to the best model based on their rating history.

Built for the **Fundamentals of AI and ML** course evaluation.

---

## 📌 Overview

CineMatch analyzes ~100,000 movie ratings from the MovieLens 100K dataset and learns patterns that allow it to:

- Recommend **personalized** movies to active users (via SVD)
- Handle **cold-start** users gracefully (via popularity baseline)
- Find **similar movies** on demand (via item-item CF)
- **Evaluate** all models on RMSE / MAE using a standard 80/20 train-test split

The system is fully **command-line driven** — no GUI is required.

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Popularity baseline** | IMDB-style weighted rating (prevents low-count movies from winning) |
| **User-User CF** | Cosine similarity between users, top-K neighbours |
| **Item-Item CF** | Cosine similarity between movies, top-K neighbours |
| **SVD (Matrix Factorization)** | 30 latent factors, mean-centred decomposition |
| **Hybrid routing** | Auto-selects best model per user based on history |
| **Cold-start handling** | Users with <5 ratings get popularity-based recommendations |
| **Similar movies** | Content-similar items via item-item similarity |
| **Evaluation** | RMSE and MAE comparison across all 4 models |
| **Model persistence** | Trained models saved to `models/` (pickle) |
| **Rich CLI** | Clean command-line interface using `argparse` |

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **scipy** — sparse matrices and truncated SVD (`svds`)
- **scikit-learn** — cosine similarity
- **argparse** — CLI parsing
- **pickle** — model persistence
- **tabulate** — tabular output (optional)

---

## 📂 Project Structure

```
AIML-cinematch/
├── data/
│   ├── raw/                        # MovieLens 100K files (u.data, u.item, ...)
│   └── processed/                  # user_item_matrix.npz + mappings.pkl
├── models/                         # Saved trained models (.pkl)
├── outputs/                        # Evaluation reports / plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Loads + merges raw files
│   ├── preprocessor.py             # Filters, encodes, builds sparse matrix
│   ├── popularity.py               # Weighted-rating recommender
│   ├── collaborative.py            # UserUserCF + ItemItemCF
│   ├── matrix_factorization.py     # SVDRecommender
│   ├── evaluator.py                # RMSE / MAE metrics
│   └── recommender.py              # HybridRecommender orchestrator
├── tests/                          # (empty — tests in root for clarity)
├── test_loader.py
├── test_preprocessor.py
├── test_popularity.py
├── test_collaborative.py
├── test_svd.py
├── test_evaluator.py
├── test_recommender.py
├── main.py                         # CLI entry point
├── requirements.txt
├── statement.md
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AIML-cinematch.git
cd AIML-cinematch
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the dataset is in place

The MovieLens 100K files (`u.data`, `u.item`, `u.user`, ...) must be in `data/raw/`.

Download from: https://grouplens.org/datasets/movielens/100k/

---

## 🚀 How to Run

All commands are run from the project root.

### Show help
```bash
python main.py --help
```

### Show dataset statistics
```bash
python main.py --info
```

### Train all models
```bash
python main.py --train
```

### Recommend movies for a user
```bash
# Auto routing (SVD for active users, Popularity for cold-start)
python main.py --recommend --user_id 1 --top_n 10

# Force a specific method
python main.py --recommend --user_id 1 --top_n 10 --method svd
python main.py --recommend --user_id 1 --top_n 10 --method user_cf
python main.py --recommend --user_id 1 --top_n 10 --method item_cf
python main.py --recommend --user_id 1 --top_n 10 --method popularity
```

### Show top-N popular movies (baseline)
```bash
python main.py --popular --top_n 10
```

### Find similar movies
```bash
python main.py --similar --movie_id 1 --top_n 5
```

### Evaluate all models (RMSE / MAE)
```bash
python main.py --evaluate
```

---

## 🧪 Testing

Each module has a dedicated sanity-check script at the project root:

```bash
python test_loader.py          # Data loader
python test_preprocessor.py    # Matrix building + split
python test_popularity.py      # Popularity recommender
python test_collaborative.py   # User-CF + Item-CF
python test_svd.py             # SVD
python test_evaluator.py       # Metrics comparison
python test_recommender.py     # End-to-end hybrid
```

All scripts exit silently on success and print progress + results along the way.

---

## 📊 Results (RMSE / MAE on MovieLens 100K Split 1)

| Model | RMSE | MAE |
| :--- | :--- | :--- |
| Popularity | 1.0037 | 0.8108 |
| UserUserCF | 0.9759 | 0.7756 |
| ItemItemCF | 0.9538 | 0.7398 |
| **SVD** 🏆 | **0.7650** | **0.5882** |

**SVD outperforms the popularity baseline by ~24% in RMSE.**

---

## 🖼️ Screenshots

_Add screenshots of the CLI output for `--recommend`, `--similar`, and `--evaluate` here._

---

## 🔭 Future Enhancements

- Add neural collaborative filtering (NCF) or ALS
- Support for a web UI (FastAPI + React)
- Incremental / online learning for new ratings
- Rich content-based features (genres, cast, directors)
- Dockerize the project for easy deployment

---

## 📄 License

Educational project — free to use and learn from.

---

## 👤 Author

**Your Name** — AIML Course Project