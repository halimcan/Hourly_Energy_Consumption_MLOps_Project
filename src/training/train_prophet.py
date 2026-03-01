import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_prophet(
    parquet_path: str,
    target_col: str,
    horizon: int = 24
):
    """
    Prophet training function (state-aware)

    Returns
    -------
    dict {
        model: trained Prophet model
        metrics: {
            mae, rmse, mape
        }
    }
    """

    # ============================
    # LOAD DATA
    # ============================
    df = pd.read_parquet(parquet_path)

    if "datetime" not in df.columns:
        raise ValueError("Prophet için 'datetime' kolonu zorunludur.")

    df = df.sort_values("datetime")

    prophet_df = df[["datetime", target_col]].rename(
        columns={
            "datetime": "ds",
            target_col: "y"
        }
    )

    # ============================
    # TRAIN / VALID SPLIT
    # ============================
    train_df = prophet_df.iloc[:-horizon]
    val_df = prophet_df.iloc[-horizon:]

    # ============================
    # MODEL
    # ============================
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(train_df)

    # ============================
    # FORECAST
    # ============================
    future = model.make_future_dataframe(
        periods=horizon,
        freq="H"
    )

    forecast = model.predict(future)

    y_true = val_df["y"].values
    y_pred = forecast.tail(horizon)["yhat"].values

    # ============================
    # METRICS
    # ============================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))
    ) * 100

    return {
        "model": model,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    }