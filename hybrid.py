
import numpy as np
import pandas as pd
import joblib

from data_utils import (
    load_movies, GENRES, RESULTS_DIR, MODELS_DIR
)
from matrix_factorization import MatrixFactorization


def metrics(actual, predicted):
    predicted = np.clip(predicted, 1, 5)
    return {
        "RMSE": float(np.sqrt(
            np.mean((actual - predicted) ** 2)
        )),
        "MAE": float(np.mean(
            np.abs(actual - predicted)
        ))
    }


def build_genre_profiles(train, movies, prior=5):
    merged = train.merge(
        movies[["item_id"] + GENRES],
        on="item_id",
        how="left"
    )

    global_mean = float(train["rating"].mean())
    profiles = {}

    for user_id, group in merged.groupby("user_id"):
        profile = {}

        for genre in GENRES:
            selected = group.loc[
                group[genre] == 1, "rating"
            ]

            profile[genre] = float(
                (selected.sum() + prior * global_mean)
                / (len(selected) + prior)
            )

        profiles[int(user_id)] = profile

    return profiles, global_mean


def predict_genres(
    df, profiles, movie_genres, global_mean
):
    predictions = []

    for user_id, item_id in zip(
        df["user_id"], df["item_id"]
    ):
        user_id = int(user_id)
        item_id = int(item_id)

        if (
            user_id not in profiles
            or item_id not in movie_genres.index
        ):
            predictions.append(global_mean)
            continue

        flags = movie_genres.loc[item_id]

        active_genres = [
            genre for genre in GENRES
            if flags[genre] == 1
        ]

        if not active_genres:
            predictions.append(global_mean)
            continue

        score = np.mean([
            profiles[user_id][genre]
            for genre in active_genres
        ])

        predictions.append(float(score))

    return np.array(predictions)


def main():
    movies = load_movies()

    train = pd.read_csv(RESULTS_DIR / "train.csv")
    val = pd.read_csv(
        RESULTS_DIR / "validation.csv"
    )
    test = pd.read_csv(RESULTS_DIR / "test.csv")

    mf = joblib.load(
        MODELS_DIR / "matrix_factorization.pkl"
    )

    profiles, global_mean = build_genre_profiles(
        train, movies
    )

    movie_genres = movies.set_index(
        "item_id"
    )[GENRES]

    val_mf = mf.predict(val)
    val_genre = predict_genres(
        val, profiles, movie_genres, global_mean
    )

    alpha_results = []

    for alpha in np.linspace(0, 1, 11):
        predictions = (
            alpha * val_mf
            + (1 - alpha) * val_genre
        )

        result = metrics(
            val["rating"].to_numpy(),
            predictions
        )

        alpha_results.append({
            "alpha": round(float(alpha), 2),
            **result
        })

    alpha_df = pd.DataFrame(alpha_results)

    best_alpha = float(
        alpha_df.loc[
            alpha_df["RMSE"].idxmin(), "alpha"
        ]
    )

    print("\n=== VALIDATION ===")
    print(alpha_df.to_string(index=False))
    print(f"\nBest alpha: {best_alpha:.2f}")

    test_mf = mf.predict(test)

    test_genre = predict_genres(
        test, profiles, movie_genres, global_mean
    )

    test_hybrid = (
        best_alpha * test_mf
        + (1 - best_alpha) * test_genre
    )

    results = []

    for name, predictions in [
        ("Matrix Factorization", test_mf),
        ("Genre Model", test_genre),
        ("Hybrid", test_hybrid)
    ]:
        result = metrics(
            test["rating"].to_numpy(),
            predictions
        )
        results.append({
            "Model": name,
            "RMSE": round(result["RMSE"], 4),
            "MAE": round(result["MAE"], 4)
        })

    results_df = pd.DataFrame(results)

    alpha_df.to_csv(
        RESULTS_DIR / "alpha_validation.csv",
        index=False
    )

    results_df.to_csv(
        RESULTS_DIR / "hybrid_results.csv",
        index=False
    )

    joblib.dump(
        {
            "profiles": profiles,
            "alpha": best_alpha,
            "global_mean": global_mean
        },
        MODELS_DIR / "hybrid_config.pkl"
    )

    print("\n=== TEST RESULTS ===")
    print(results_df.to_string(index=False))
    print("\nConfiguratia hibrida a fost salvata.")


if __name__ == "__main__":
    main()
