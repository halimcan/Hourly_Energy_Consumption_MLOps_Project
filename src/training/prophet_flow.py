import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet


def train_prophet(data_path: str, target_col: str):
    """
    Prophet training & evaluation.
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

    if "Datetime" not in df.columns:
        raise ValueError("Prophet requires 'Datetime' column")

    # -------------------------
    # 2️⃣ Prepare Prophet format
    # -------------------------
    prophet_df = df[["Datetime", target_col]].rename(
        columns={
            "Datetime": "ds",
            target_col: "y"
        }
    )

    prophet_df = prophet_df.sort_values("ds")

    # -------------------------
    # 3️⃣ Train / Test split
    # -------------------------
    if len(prophet_df) <= 24:
        raise ValueError(
            "Not enough data to create train/test split (need > 24 rows)"
        )

    train_df = prophet_df.iloc[:-24]
    test_df = prophet_df.iloc[-24:]

    # -------------------------
    # 4️⃣ Model
    # -------------------------
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(train_df)

    # -------------------------
    # 5️⃣ Forecast
    # -------------------------
    future = model.make_future_dataframe(
        periods=24,
        freq="H"
    )

    forecast = model.predict(future)

    predictions = forecast.tail(24)["yhat"].values
    y_test = test_df["y"].values

    # -------------------------
    # 6️⃣ Evaluate
    # -------------------------
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mape = np.mean(
        np.abs((y_test - predictions) / y_test)
    ) * 100

    return {
        "model_name": "prophet",
        "model": model,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
        },
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
    }