
import pandas as pd
import joblib
import numpy as np

from adaptive_hybrid import adaptive_alpha

from data_utils import (
    load_movies,
    load_ratings,
    GENRES,
    RESULTS_DIR,
    MODELS_DIR
)

from matrix_factorization import MatrixFactorization


class MovieRecommender:

    def __init__(self):


        self.movies = load_movies()
        self.ratings = load_ratings()

        self.train = pd.read_csv(
            RESULTS_DIR / "train.csv"
        )

        self.mf = joblib.load(
            MODELS_DIR / "matrix_factorization.pkl"
        )

        self.hybrid = joblib.load(
            MODELS_DIR / "hybrid_config.pkl"
        )

        self.baselines = joblib.load(
            MODELS_DIR / "baselines.pkl"
        )

        self.profiles = self.hybrid["profiles"]
        self.alpha = self.hybrid["alpha"]
        self.global_mean = self.hybrid["global_mean"]

        self.movie_scores = self.baselines[
            "movie_scores"
        ]


        self.adaptive_config = joblib.load(
            MODELS_DIR / "adaptive_config.pkl"
        )

        self.user_counts = (
            self.train.groupby("user_id")
            .size()
            .to_dict()
        )


        self.movie_genres = self.movies.set_index(
            "item_id"
        )[GENRES]

    def genre_score(self, user_id, item_id):

        profile = self.profiles.get(
            int(user_id)
        )

        if profile is None:
            return self.global_mean

        if item_id not in self.movie_genres.index:
            return self.global_mean

        flags = self.movie_genres.loc[item_id]

        active = [
            genre
            for genre in GENRES
            if flags[genre] == 1
        ]

        if not active:
            return self.global_mean

        return float(
            np.mean([
                profile[genre]
                for genre in active
            ])
        )

    def get_adaptive_alpha(self, user_id):
        """
        Calculeaza ponderea MF pentru un utilizator
        folosind configuratia optimizata pe validation.
        """

        user_id = int(user_id)

        n_ratings = self.user_counts.get(
            user_id, 0
        )

        return float(
            adaptive_alpha(
                n_ratings=n_ratings,
                alpha_min=self.adaptive_config[
                    "alpha_min"
                ],
                alpha_max=self.adaptive_config[
                    "alpha_max"
                ],
                k=self.adaptive_config["k"]
            )
        )

    def recommend_existing(
        self,
        user_id,
        n=10,
        method="Hybrid"
    ):

        user_id = int(user_id)

        if user_id not in self.mf.known_users:
            raise ValueError(
                "Utilizatorul nu exista "
                "in setul de antrenare."
            )

        valid_methods = [
            "Matrix Factorization",
            "Genre Model",
            "Hybrid",
            "Hybrid Adaptive"
        ]

        if method not in valid_methods:
            raise ValueError(
                f"Metoda necunoscuta: {method}"
            )

        seen = set(
            self.ratings.loc[
                self.ratings["user_id"] == user_id,
                "item_id"
            ]
        )

        candidates = self.movies.loc[
            ~self.movies["item_id"].isin(seen)
        ].copy()

        if method == "Hybrid Adaptive":
            alpha = self.get_adaptive_alpha(
                user_id
            )
        else:
            alpha = self.alpha

        scores = []

        for item_id in candidates["item_id"]:

            item_id = int(item_id)

            mf_score = self.mf.predict_one(
                user_id,
                item_id
            )

            genre_score = self.genre_score(
                user_id,
                item_id
            )

            if method == "Matrix Factorization":

                score = mf_score

            elif method == "Genre Model":

                score = genre_score


            elif method == "Hybrid":

                score = (
                    self.alpha * mf_score
                    + (1 - self.alpha) * genre_score
                )

            elif method == "Hybrid Adaptive":

                score = (
                    alpha * mf_score
                    + (1 - alpha) * genre_score
                )

            scores.append(
                float(np.clip(score, 1, 5))
            )

        candidates["score"] = scores

        return (
            candidates.sort_values(
                ["score", "item_id"],
                ascending=[False, True]
            )
            .head(n)
            .reset_index(drop=True)
        )

    def recommend_new(
        self,
        preferred_genres,
        n=10
    ):

        if not preferred_genres:
            return pd.DataFrame()

        invalid = (
            set(preferred_genres)
            - set(GENRES)
        )

        if invalid:
            raise ValueError(
                f"Genuri necunoscute: {invalid}"
            )

        candidates = self.movies.copy()

        candidates["genre_match"] = (
            candidates[preferred_genres].sum(axis=1)
            / len(preferred_genres)
        )

        candidates = candidates.loc[
            candidates["genre_match"] > 0
        ].copy()

        candidates["movie_mean"] = (
            candidates["item_id"]
            .map(self.movie_scores)
            .fillna(self.global_mean)
        )

        quality = (
            candidates["movie_mean"] - 1
        ) / 4

        candidates["score"] = (
            0.7 * candidates["genre_match"]
            + 0.3 * quality
        )

        return (
            candidates.sort_values(
                [
                    "score",
                    "movie_mean",
                    "item_id"
                ],
                ascending=[
                    False,
                    False,
                    True
                ]
            )
            .head(n)
            .reset_index(drop=True)
        )
