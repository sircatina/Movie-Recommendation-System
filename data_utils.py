
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ml-100k"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

GENRES = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

MOVIE_COLUMNS = [
    "item_id", "title", "release_date",
    "video_release_date", "imdb_url"
] + GENRES


def load_ratings():
    return pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=[
            "user_id", "item_id",
            "rating", "timestamp"
        ]
    )


def load_movies():
    return pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        names=MOVIE_COLUMNS,
        encoding="latin-1"
    )
