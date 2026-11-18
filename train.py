
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from data_utils import (
    load_ratings, RESULTS_DIR, MODELS_DIR
)
from matrix_factorization import MatrixFactorization


def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, 5)

    return {
        "RMSE": round(float(np.sqrt(
            mean_squared_error(y_true, y_pred)
        )), 4),
        "MAE": round(float(
            mean_absolute_error(y_true, y_pred)
        ), 4)
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    ratings = load_ratings()

    train_val, test = train_test_split(
        ratings,
        test_size=0.10,
        random_state=42
    )

    train, val = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=42
    )

    print("=== IMPARTIRE DATASET ===")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    global_mean = float(train["rating"].mean())

    global_predictions = np.full(
        len(test), global_mean
    )

    movie_stats = train.groupby("item_id")[
        "rating"
    ].agg(["mean", "count"])

    shrinkage = 10

    movie_stats["score"] = (
        movie_stats["count"] * movie_stats["mean"]
        + shrinkage * global_mean
    ) / (movie_stats["count"] + shrinkage)

    movie_scores = movie_stats["score"].to_dict()

    movie_predictions = (
        test["item_id"]
        .map(movie_scores)
        .fillna(global_mean)
        .to_numpy()
    )

    model = MatrixFactorization(
        n_users=int(ratings["user_id"].max()),
        n_items=int(ratings["item_id"].max()),
        n_factors=30,
        learning_rate=0.01,
        reg=0.05,
        epochs=20,
        seed=42
    )

    print("\n=== ANTRENARE ===")
    model.fit(train, val)

    mf_predictions = model.predict(test)

    results = []

    for name, predictions in [
        ("Global Mean", global_predictions),
        ("Movie Mean", movie_predictions),
        ("Matrix Factorization", mf_predictions)
    ]:
        result = evaluate(
            test["rating"], predictions
        )
        results.append({
            "Model": name,
            **result
        })

    results_df = pd.DataFrame(results)

    train.to_csv(
        RESULTS_DIR / "train.csv", index=False
    )
    val.to_csv(
        RESULTS_DIR / "validation.csv", index=False
    )
    test.to_csv(
        RESULTS_DIR / "test.csv", index=False
    )

    results_df.to_csv(
        RESULTS_DIR / "baseline_results.csv",
        index=False
    )

    joblib.dump(
        model,
        MODELS_DIR / "matrix_factorization.pkl"
    )

    joblib.dump(
        {
            "global_mean": global_mean,
            "movie_scores": movie_scores
        },
        MODELS_DIR / "baselines.pkl"
    )

    print("\n=== REZULTATE FINALE ===")
    print(results_df.to_string(index=False))
    print("\nModelele au fost salvate.")


if __name__ == "__main__":
    main()
