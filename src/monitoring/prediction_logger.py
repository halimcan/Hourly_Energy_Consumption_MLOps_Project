from pathlib import Path
import polars as pl
from datetime import datetime


LOG_PATH = Path("logs/predictions.parquet")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(
    state: str,
    model_type: str,
    prediction: float,
    features: dict | None = None,
    target: float | None = None
):
    """
    Append prediction log entry.
    """

    row = {
        "timestamp": datetime.utcnow(),
        "state": state,
        "model_type": model_type,
        "prediction": prediction,
        "target": target,
    }

    # feature snapshot (flatten)
    if features:
        for k, v in features.items():
            row[f"f_{k}"] = v

    df_new = pl.DataFrame([row])

    if LOG_PATH.exists():
        df_old = pl.read_parquet(LOG_PATH)
        df_all = pl.concat([df_old, df_new])
    else:
        df_all = df_new

    df_all.write_parquet(LOG_PATH)