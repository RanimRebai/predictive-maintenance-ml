from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import add_rolling_features, COLUMNS, RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"


def read_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )


def main():
    # Load training data and calculate RUL targets.
    train_df = read_data(TRAIN_PATH)

    max_cycles = train_df.groupby("unit")["cycle"].transform("max")
    train_df["RUL"] = max_cycles - train_df["cycle"]

    train_df = add_rolling_features(train_df)

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    setting_cols = [f"setting_{i}" for i in range(1, 4)]

    rolling_cols = [
        f"sensor_{i}_rolling_mean_{window}"
        for window in [5, 10]
        for i in range(1, 22)
    ]

    feature_cols = sensor_cols + setting_cols + rolling_cols

    # Train the selected model on all training engines.
    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training model on the full training dataset...")
    model.fit(train_df[feature_cols], train_df["RUL"])

    # Load the separate test sensor sequences.
    test_df = read_data(TEST_PATH)
    test_df = add_rolling_features(test_df)

    # Get the official RUL labels for the test engines.
    true_rul = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    # Predict RUL at the final observed cycle of each engine.
    final_rows = (
        test_df.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
    )

    if len(final_rows) != len(true_rul):
        raise ValueError(
            f"Found {len(final_rows)} test engines but "
            f"{len(true_rul)} RUL labels."
        )

    predictions = model.predict(final_rows[feature_cols])

    mae = mean_absolute_error(true_rul, predictions)
    rmse = np.sqrt(mean_squared_error(true_rul, predictions))
    r2 = r2_score(true_rul, predictions)

    print("\nPredicta — Official FD001 Test Evaluation")
    print("-----------------------------------------")
    print(f"Test engines: {len(final_rows)}")
    print(f"MAE:  {mae:.3f} cycles")
    print(f"RMSE: {rmse:.3f} cycles")
    print(f"R²:   {r2:.3f}")


if __name__ == "__main__":
    main()