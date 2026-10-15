
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"/"raw" / "train_FD001.txt"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_rul.joblib"

# Reproducibility
RANDOM_STATE = 42


# NASA C-MAPSS FD001 dataset columns
COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data():
    """Load engine sensor data and calculate remaining useful life."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Check that train_FD001.txt is inside the data folder."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]

    return df


def add_rolling_features(df):
    """Add 5-cycle and 10-cycle rolling sensor averages."""
    df = df.sort_values(["unit", "cycle"]).copy()

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

    for window in [5, 10]:
        rolling = (
            df.groupby("unit")[sensor_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling.columns = [
            f"{col}_rolling_mean_{window}"
            for col in sensor_cols
        ]

        df = df.join(rolling)

    return df


def train_model():
    """Train, evaluate, and save the Random Forest RUL model."""
    df = add_rolling_features(load_data())

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    setting_cols = [f"setting_{i}" for i in range(1, 4)]
    rolling_cols = [
        f"sensor_{i}_rolling_mean_{window}"
        for window in [5, 10]
        for i in range(1, 22)
    ]
    feature_cols = sensor_cols + setting_cols + rolling_cols

    # Split by engine to avoid putting the same engine in both sets.
    engine_ids = df["unit"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_df = df[df["unit"].isin(train_ids)]
    val_df = df[df["unit"].isin(val_ids)]

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]
    X_val = val_df[feature_cols]
    y_val = val_df["RUL"]

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    print("\nValidation results")
    print(f"MAE:  {mae:.3f} cycles")
    print(f"RMSE: {rmse:.3f} cycles")
    print(f"R²:   {r2:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()