
import numpy as np
import pandas as pd
import joblib

from data_utils import (
    load_movies,
    GENRES,
    RESULTS_DIR,
    MODELS_DIR
)

from hybrid import predict_genres, metrics

def adaptive_alpha(
    n_ratings,
    alpha_min=0.5,
    alpha_max=0.95,
    k=20
):
    """
    Calculeaza ponderea MF in functie de
    numarul evaluarilor utilizatorului.

    Pentru istoric redus -> pondere MF mai mica.
    Pentru istoric bogat -> pondere MF mai mare.
    """

    return (
        alpha_min
        + (alpha_max - alpha_min)
        * n_ratings / (n_ratings + k)
    )


def adaptive_predictions(
    df,
    mf_predictions,
    genre_predictions,
    user_counts,
    alpha_min,
    alpha_max,
    k
):
    """
    Combina predictiile MF si Genre Model
    folosind o pondere specifica utilizatorului.
    """

    counts = (
        df["user_id"]
        .map(user_counts)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    alphas = adaptive_alpha(
        counts,
        alpha_min,
        alpha_max,
        k
    )

    predictions = (
        alphas * mf_predictions
        + (1 - alphas) * genre_predictions
    )

    return np.clip(predictions, 1, 5)


def main():

    train = pd.read_csv(
        RESULTS_DIR / "train.csv"
    )

    val = pd.read_csv(
        RESULTS_DIR / "validation.csv"
    )

    test = pd.read_csv(
        RESULTS_DIR / "test.csv"
    )

    movies = load_movies()

    mf = joblib.load(
        MODELS_DIR / "matrix_factorization.pkl"
    )

    config = joblib.load(
        MODELS_DIR / "hybrid_config.pkl"
    )

    profiles = config["profiles"]
    global_mean = config["global_mean"]
    fixed_alpha = config["alpha"]

    # Aceeasi structura ca in hybrid.py
    movie_genres = movies.set_index(
        "item_id"
    )[GENRES]

    user_counts = (
        train.groupby("user_id")
        .size()
        .to_dict()
    )

    print(
        "\n=== DISTRIBUTIA NUMARULUI DE EVALUARI ==="
    )

    print(
        pd.Series(user_counts).describe()
    )

    mf_val = mf.predict(val)

    genre_val = predict_genres(
        val,
        profiles,
        movie_genres,
        global_mean
    )

    experiments = []

    for alpha_min in [0.3, 0.5, 0.7]:

        for alpha_max in [0.9, 0.95, 1.0]:

            for k in [5, 10, 20, 50, 100]:

                predictions = adaptive_predictions(
                    df=val,
                    mf_predictions=mf_val,
                    genre_predictions=genre_val,
                    user_counts=user_counts,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    k=k
                )

                result = metrics(
                    val["rating"].to_numpy(),
                    predictions
                )

                experiments.append({
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "k": k,
                    "RMSE": result["RMSE"],
                    "MAE": result["MAE"]
                })

    experiments_df = pd.DataFrame(
        experiments
    ).sort_values(
        ["RMSE", "MAE"]
    ).reset_index(drop=True)

    experiments_df.to_csv(
        RESULTS_DIR / "adaptive_validation.csv",
        index=False
    )

    best = experiments_df.iloc[0]

    alpha_min = float(best["alpha_min"])
    alpha_max = float(best["alpha_max"])
    k = float(best["k"])

    print("\n=== BEST ADAPTIVE CONFIG ===")
    print(best.to_string())

    mf_test = mf.predict(test)

    genre_test = predict_genres(
        test,
        profiles,
        movie_genres,
        global_mean
    )

    adaptive_test = adaptive_predictions(
        df=test,
        mf_predictions=mf_test,
        genre_predictions=genre_test,
        user_counts=user_counts,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        k=k
    )

    fixed_test = (
        fixed_alpha * mf_test
        + (1 - fixed_alpha) * genre_test
    )

    results = []

    for name, predictions in [
        ("Matrix Factorization", mf_test),
        ("Genre Model", genre_test),
        ("Hybrid Fixed", fixed_test),
        ("Hybrid Adaptive", adaptive_test)
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

    results_df.to_csv(
        RESULTS_DIR / "adaptive_results.csv",
        index=False
    )

    joblib.dump(
        {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "k": k,
            "selection_metric": "RMSE",
            "fixed_alpha": float(fixed_alpha)
        },
        MODELS_DIR / "adaptive_config.pkl"
    )

    print("\n=== TEST RESULTS ===")

    print(
        results_df.to_string(index=False)
    )

    print(
        "\nConfiguratia adaptiva a fost salvata."
    )


if __name__ == "__main__":
    main()
