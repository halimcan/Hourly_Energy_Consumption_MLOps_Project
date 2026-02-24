import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from src.models.baseline_model import BaselineModel


def train_baseline(data_path: str, target_col: str):
    """
    Baseline model training & evaluation.
    Compatible with unified model selection pipeline.
    """

    # -------------------------
    # 1️⃣ Load data
    # -------------------------
    df = pd.read_parquet(data_path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # -------------------------
    # 2️⃣ Feature temizliği
    # -------------------------
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # -------------------------
    # 3️⃣ Train / Test split
    # -------------------------
    if len(df) <= 24:
        raise ValueError(
            "Not enough data to create train/test split (need > 24 rows)"
        )

    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    # -------------------------
    # 4️⃣ Model
    # -------------------------
    model = BaselineModel(target_col=target_col)

    # Baseline genelde naive olduğu için fit mantıksal
    model.fit(train_df)

    # -------------------------
    # 5️⃣ Forecast
    # -------------------------
    predictions = model.forecast_next_24(train_df)

    y_test = test_df[target_col]

    # -------------------------
    # 6️⃣ Evaluate
    # -------------------------
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # MAPE
    mape = np.mean(
        np.abs((y_test - predictions) / y_test)
    ) * 100

    return {
        "model_name": "baseline",
        "model": model,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
        },
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
    }