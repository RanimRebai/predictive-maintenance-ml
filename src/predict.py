
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_rul.joblib"


def load_model():
    """Load the trained model and its expected feature names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train.py first."
        )

    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_cols"]


def predict_rul(sensor_history):
    """
    Predict RUL from one engine's sensor history.

    sensor_history must be a pandas DataFrame containing:
    unit, cycle, setting_1..setting_3, and sensor_1..sensor_21.
    """
    model, feature_cols = load_model()

    required_cols = (
        ["unit", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    missing = set(required_cols) - set(sensor_history.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if sensor_history.empty:
        raise ValueError("Sensor history cannot be empty.")

    history = sensor_history[required_cols].copy()
    history = history.sort_values("cycle")

    if history["unit"].nunique() != 1:
        raise ValueError("Provide sensor history for exactly one engine.")

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

    for window in [5, 10]:
        rolling = (
            history[sensor_cols]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        rolling.columns = [
            f"{col}_rolling_mean_{window}"
            for col in sensor_cols
        ]
        history = history.join(rolling)

    latest = history.iloc[[-1]]
    prediction = float(model.predict(latest[feature_cols])[0])

    return max(0.0, prediction)


if __name__ == "__main__":
    COLUMNS = (
        ["unit", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    DATA_PATH = PROJECT_ROOT / "data" /"raw" / "train_FD001.txt"

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    # Use the first 30 cycles of engine 1 for a simple test.
    engine_history = df[
        (df["unit"] == 1) & (df["cycle"] <= 30)
    ].copy()

    prediction = predict_rul(engine_history)

    print(f"Engine: 1")
    print(f"Latest observed cycle: {engine_history['cycle'].max()}")
    print(f"Predicted remaining useful life: {prediction:.1f} cycles")