import pandas as pd
from datetime import datetime


def predict(model, model_type: str, data):
    """
    Unified prediction interface

    Parameters
    ----------
    model : trained model object
    model_type : str
        baseline | prophet | xgboost | lgbm | sklearn
    data :
        - baseline   -> ignored
        - prophet    -> datetime OR None
        - others     -> feature vector (np.array / pd.DataFrame)
    """

    model_type = model_type.lower()

    # ============================
    # BASELINE
    # ============================
    if model_type == "baseline":
        # always returns 24h forecast
        return model.forecast_next_24(None)

    # ============================
    # PROPHET
    # ============================
    if model_type == "prophet":

        # 1️⃣ Tek datetime prediction
        if isinstance(data, datetime):
            df = pd.DataFrame({"ds": [data]})
            forecast = model.predict(df)
            return forecast["yhat"].values

        # 2️⃣ Next-24 forecast (opsiyonel)
        if data is None:
            future = model.make_future_dataframe(
                periods=24,
                freq="H"
            )
            forecast = model.predict(future)
            return forecast.tail(24)["yhat"].values

        raise ValueError(
            "Prophet için data datetime veya None olmalıdır."
        )

    # ============================
    # FEATURE-BASED MODELS
    # ============================
    try:
        return model.predict(data)
    except Exception as e:
        raise RuntimeError(
            f"Predict failed for model_type={model_type}: {str(e)}"
        )