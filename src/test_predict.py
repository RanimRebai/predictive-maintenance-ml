
import pandas as pd
import pytest

from predict import predict_rul


def make_sensor_history(num_cycles=15):
    """Create sample sensor data for one engine."""
    rows = []

    for cycle in range(1, num_cycles + 1):
        row = {
            "unit": 1,
            "cycle": cycle,
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        }

        for i in range(1, 22):
            row[f"sensor_{i}"] = 100.0 + i + cycle * 0.1

        rows.append(row)

    return pd.DataFrame(rows)


def test_prediction_returns_nonnegative_number():
    history = make_sensor_history()

    prediction = predict_rul(history)

    assert isinstance(prediction, float)
    assert prediction >= 0


def test_empty_history_is_rejected():
    history = make_sensor_history().iloc[0:0]

    with pytest.raises(ValueError, match="cannot be empty"):
        predict_rul(history)


def test_multiple_engines_are_rejected():
    history = make_sensor_history()
    second_engine = history.copy()
    second_engine["unit"] = 2
    combined = pd.concat([history, second_engine], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one engine"):
        predict_rul(combined)


def test_missing_columns_are_rejected():
    history = make_sensor_history().drop(columns=["sensor_1"])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_rul(history)