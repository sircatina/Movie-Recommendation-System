
import numpy as np
import pandas as pd
import joblib

from data_utils import (
    load_movies,
    GENRES,
    RESULTS_DIR,
    MODELS_DIR
)

from hybrid import predict_genres
from adaptive_hybrid import adaptive_alpha
from matrix_factorization import MatrixFactorization


K = 10
RELEVANCE_THRESHOLD = 4


def precision_at_k(relevance, k=10):
    return float(np.sum(relevance[:k]) / k)


def recall_at_k(relevance, k=10):
    total_relevant = np.sum(relevance)

    if total_relevant == 0:
        return np.nan

    return float(
        np.sum(relevance[:k]) / total_relevant
    )


def ndcg_at_k(relevance, k=10):
    relevance = np.asarray(
        relevance,
        dtype=float
    )

    discounts = np.log2(
        np.arange(2, k + 2)
    )

    dcg = np.sum(
        relevance[:k] / discounts
    )

    ideal = np.sort(relevance)[::-1]

    idcg = np.sum(
        ideal[:k] / discounts
    )

    if idcg == 0:
        return np.nan

    return float(dcg / idcg)


def evaluate_user(
    user_df,
    predictions,
    method,
    k=10
):
    ranked = user_df[
        ["item_id", "rating"]
    ].copy()

    ranked["score"] = predictions

    ranked = ranked.sort_values(
        ["score", "item_id"],
        ascending=[False, True]
    )

    relevance = (
        ranked["rating"].to_numpy()
        >= RELEVANCE_THRESHOLD
    ).astype(int)

    return {
        "user_id": int(
            user_df["user_id"].iloc[0]
        ),
        "model": method,
        "n_candidates": len(ranked),
        "n_relevant": int(relevance.sum()),
        "Precision@10": precision_at_k(
            relevance, k
        ),
        "Recall@10": recall_at_k(
            relevance, k
        ),
        "NDCG@10": ndcg_at_k(
            relevance, k
        )
    }


def main():
    train = pd.read_csv(
        RESULTS_DIR / "train.csv"
    )

    test = pd.read_csv(
        RESULTS_DIR / "test.csv"
    )

    movies = load_movies()

    mf = joblib.load(
        MODELS_DIR / "matrix_factorization.pkl"
    )

    hybrid_config = joblib.load(
        MODELS_DIR / "hybrid_config.pkl"
    )

    adaptive_config = joblib.load(
        MODELS_DIR / "adaptive_config.pkl"
    )

    profiles = hybrid_config["profiles"]
    global_mean = hybrid_config["global_mean"]
    fixed_alpha = hybrid_config["alpha"]

    movie_genres = movies.set_index(
        "item_id"
    )[GENRES]

    user_counts = (
        train.groupby("user_id")
        .size()
        .to_dict()
    )

    eligible_users = []

    for user_id, group in test.groupby("user_id"):
        if (
            len(group) >= K
            and (group["rating"] >=
                 RELEVANCE_THRESHOLD).any()
            and int(user_id) in mf.known_users
        ):
            eligible_users.append(int(user_id))

    print(
        f"Utilizatori totali in test: "
        f"{test['user_id'].nunique()}"
    )

    print(
        f"Utilizatori eligibili: "
        f"{len(eligible_users)}"
    )

    if not eligible_users:
        raise RuntimeError(
            "Nu exista utilizatori eligibili "
            "pentru protocolul de evaluare."
        )

    eligible_test = test[
        test["user_id"].isin(eligible_users)
    ].copy()

    mf_predictions = mf.predict(
        eligible_test
    )

    genre_predictions = predict_genres(
        eligible_test,
        profiles,
        movie_genres,
        global_mean
    )

    fixed_predictions = (
        fixed_alpha * mf_predictions
        + (1 - fixed_alpha) * genre_predictions
    )

    counts = (
        eligible_test["user_id"]
        .map(user_counts)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    alphas = adaptive_alpha(
        n_ratings=counts,
        alpha_min=adaptive_config["alpha_min"],
        alpha_max=adaptive_config["alpha_max"],
        k=adaptive_config["k"]
    )

    adaptive_predictions = (
        alphas * mf_predictions
        + (1 - alphas) * genre_predictions
    )

    model_predictions = {
        "Matrix Factorization": mf_predictions,
        "Genre Model": genre_predictions,
        "Hybrid Fixed": fixed_predictions,
        "Hybrid Adaptive": adaptive_predictions
    }

    detailed_results = []

    for method, predictions in model_predictions.items():
        scored = eligible_test.copy()

        scored["predicted_score"] = np.clip(
            predictions, 1, 5
        )

        for user_id, group in scored.groupby("user_id"):
            result = evaluate_user(
                user_df=group,
                predictions=group[
                    "predicted_score"
                ].to_numpy(),
                method=method,
                k=K
            )

            detailed_results.append(result)

    details_df = pd.DataFrame(
        detailed_results
    )

    summary_df = (
        details_df.groupby("model")
        .agg(
            users=("user_id", "nunique"),
            mean_candidates=(
                "n_candidates", "mean"
            ),
            mean_relevant=(
                "n_relevant", "mean"
            ),
            **{
                "Precision@10": (
                    "Precision@10", "mean"
                ),
                "Recall@10": (
                    "Recall@10", "mean"
                ),
                "NDCG@10": (
                    "NDCG@10", "mean"
                )
            }
        )
        .reset_index()
        .sort_values(
            "NDCG@10",
            ascending=False
        )
    )

    details_df.to_csv(
        RESULTS_DIR / "ranking_per_user.csv",
        index=False
    )

    summary_df.to_csv(
        RESULTS_DIR / "ranking_results.csv",
        index=False
    )

    print("\n=== RANKING RESULTS ===")

    print(
        summary_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\nRezultatele au fost salvate."
    )


if __name__ == "__main__":
    main()
