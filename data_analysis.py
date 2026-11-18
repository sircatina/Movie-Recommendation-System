
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ml-100k"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

ratings = pd.read_csv(
    DATA_DIR / "u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
    encoding="latin-1"
)

genres = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

columns = [
    "item_id", "title", "release_date",
    "video_release_date", "imdb_url"
] + genres

movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    names=columns,
    encoding="latin-1"
)

# 4. Combinarea datelor
df = ratings.merge(
    movies,
    on="item_id",
    how="left"
)

# 5. Statistici
print("\n=== MOVIELENS 100K ===")
print(f"Număr ratinguri: {len(ratings)}")
print(f"Număr utilizatori: {ratings['user_id'].nunique()}")
print(f"Număr filme evaluate: {ratings['item_id'].nunique()}")
print(f"Număr filme în catalog: {len(movies)}")

print("\nDistribuția ratingurilor:")
print(ratings["rating"].value_counts().sort_index())

print("\nValori lipsă:")
print(df[["user_id", "item_id", "rating", "title"]].isna().sum())

# 6. Densitatea matricei utilizator-film
n_users = ratings["user_id"].nunique()
n_movies = len(movies)

density = len(ratings) / (n_users * n_movies)

print(f"\nDensitate: {density:.2%}")
print(f"Sparsitate: {1 - density:.2%}")

# 7. Top 10 filme după numărul de evaluări
top_movies = (
    df.groupby(["item_id", "title"])
    .agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    .sort_values("num_ratings", ascending=False)
    .head(10)
)

print("\nTop 10 filme după popularitate:")
print(top_movies.to_string())

# 8. Salvarea datelor
df.to_csv(
    RESULTS_DIR / "movielens_processed.csv",
    index=False
)

print("\nDatele au fost salvate în results/movielens_processed.csv")
