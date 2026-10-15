
from pathlib import Path

import pandas as pd

from predict import predict_rul


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"/"raw" / "train_FD001.txt"

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    available_engines = sorted(df["unit"].unique())

    print("\nPredicta — Predictive Maintenance")
    print("--------------------------------")

    try:
        engine_id = int(input("Enter engine ID (1–100): "))
        if engine_id not in available_engines:
            print("Invalid engine ID. Choose an ID from the available dataset.")
            return

        engine_history = df[df["unit"] == engine_id].copy()

        max_cycle = int(engine_history["cycle"].max())
        print(f"Available history: 1–{max_cycle} cycles")

        cutoff = int(input("Predict using history through which cycle? "))

        if cutoff < 1 or cutoff > max_cycle:
            print(f"Please enter a cycle between 1 and {max_cycle}.")
            return

        history = engine_history[
            engine_history["cycle"] <= cutoff
        ].copy()

        prediction = predict_rul(history)

        print("\nPrediction result")
        print("-----------------")
        print(f"Engine ID: {engine_id}")
        print(f"Latest observed cycle: {cutoff}")
        print(f"Estimated remaining useful life: {prediction:.1f} cycles")

    except ValueError:
        print("Please enter valid whole numbers for the engine ID and cycle.")


if __name__ == "__main__":
    main()