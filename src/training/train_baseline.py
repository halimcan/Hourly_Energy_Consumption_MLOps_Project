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
    Primary metric: RMSE
    Secondary metrics: MAE, MAPE
    """

    # -------------------------------------------------
    # 1️⃣ Load data
    # -------------------------------------------------
    df = pd.read_parquet(data_path)

    # -------------------------------------------------
    # 2️⃣ Feature cleanup (metadata çıkar)
    # -------------------------------------------------
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # -------------------------------------------------
    # 3️⃣ Train / Test split
    # Son 24 saat test
    # -------------------------------------------------
    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    y_true = test_df[target_col].values

    # -------------------------------------------------
    # 4️⃣ Model
    # -------------------------------------------------
    model = BaselineModel(target_col=target_col)

    # Baseline model fit gerektirmez
    model.fit(train_df)

    # -------------------------------------------------
    # 5️⃣ Forecast
    # -------------------------------------------------
    y_pred = model.forecast_next_24(train_df)
    y_pred = np.array(y_pred)

    # -------------------------------------------------
    # 6️⃣ Metrics
    # -------------------------------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # MAPE (0 bölme koruması)
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = (
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ).mean() * 100
    else:
        mape = None

    # -------------------------------------------------
    # 7️⃣ Result
    # -------------------------------------------------
    return {
        "model_name": "baseline",
        "model": model,
        "primary_metric": "rmse",
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape) if mape is not None else None
        }
    }