
import numpy as np
from sklearn.metrics import mean_squared_error


class MatrixFactorization:
    def __init__(
        self, n_users, n_items, n_factors=30,
        learning_rate=0.01, reg=0.05,
        epochs=20, seed=42
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.user_factors = rng.normal(
            0, 0.1, (n_users + 1, n_factors)
        )
        self.item_factors = rng.normal(
            0, 0.1, (n_items + 1, n_factors)
        )
        self.user_bias = np.zeros(n_users + 1)
        self.item_bias = np.zeros(n_items + 1)

        self.global_mean = 0.0
        self.known_users = set()
        self.known_items = set()

    def predict_one(self, user, item):
        user, item = int(user), int(item)

        if (
            user not in self.known_users
            or item not in self.known_items
        ):
            return float(self.global_mean)

        prediction = (
            self.global_mean
            + self.user_bias[user]
            + self.item_bias[item]
            + np.dot(
                self.user_factors[user],
                self.item_factors[item]
            )
        )
        return float(np.clip(prediction, 1, 5))

    def predict(self, dataframe):
        return np.array([
            self.predict_one(u, i)
            for u, i in zip(
                dataframe["user_id"],
                dataframe["item_id"]
            )
        ])

    def fit(self, train_df, val_df):
        self.global_mean = float(
            train_df["rating"].mean()
        )
        self.known_users = set(train_df["user_id"])
        self.known_items = set(train_df["item_id"])

        samples = train_df[
            ["user_id", "item_id", "rating"]
        ].to_numpy().copy()

        rng = np.random.default_rng(self.seed)
        best_rmse = float("inf")
        best_state = None
        no_improvement = 0

        for epoch in range(self.epochs):
            rng.shuffle(samples)

            for user, item, rating in samples:
                user, item = int(user), int(item)

                prediction = (
                    self.global_mean
                    + self.user_bias[user]
                    + self.item_bias[item]
                    + np.dot(
                        self.user_factors[user],
                        self.item_factors[item]
                    )
                )
                error = rating - prediction

                pu = self.user_factors[user].copy()
                qi = self.item_factors[item].copy()

                self.user_bias[user] += self.lr * (
                    error - self.reg * self.user_bias[user]
                )
                self.item_bias[item] += self.lr * (
                    error - self.reg * self.item_bias[item]
                )
                self.user_factors[user] += self.lr * (
                    error * qi - self.reg * pu
                )
                self.item_factors[item] += self.lr * (
                    error * pu - self.reg * qi
                )

            val_rmse = float(np.sqrt(
                mean_squared_error(
                    val_df["rating"],
                    self.predict(val_df)
                )
            ))

            print(
                f"Epoch {epoch + 1:02d} "
                f"- Validation RMSE: {val_rmse:.4f}"
            )

            if val_rmse < best_rmse - 0.0001:
                best_rmse = val_rmse
                no_improvement = 0
                best_state = (
                    self.user_factors.copy(),
                    self.item_factors.copy(),
                    self.user_bias.copy(),
                    self.item_bias.copy()
                )
            else:
                no_improvement += 1

            if no_improvement >= 4:
                print("Early stopping")
                break

        if best_state is not None:
            (
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias
            ) = best_state

        return self
