import polars as pl
import numpy as np


def compute_persistence_rmse_last_24(window_df: pl.DataFrame) -> float | None:
    """
    Persistence baseline (son 24 saat):

    y_hat(t) = y(t-24)

    Son 24 saatin gerçek değerleri ile
    bir önceki günün değerlerini karşılaştırır.
    """

    if window_df.height < 48:
        return None

    df = window_df.sort("Datetime").to_pandas()

    # Son 24 saat gerçek değer
    y_true = df["target"].values[-24:]

    # Bir önceki günün aynı saatleri
    y_pred = df["target"].values[-48:-24]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return float(rmse)