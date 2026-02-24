import numpy as np


def calculate_business_metrics(
    y_true,
    y_pred,
    unit_cost_over: float = 1.0,
    unit_cost_under: float = 3.0,
    peak_quantile: float = 0.9,
):
    """
    Business-oriented metrics for energy forecasting.

    Parameters
    ----------
    y_true : array-like
        Actual consumption
    y_pred : array-like
        Forecasted consumption
    unit_cost_over : float
        Cost of overproduction per unit
    unit_cost_under : float
        Cost of underproduction per unit (usually higher)
    peak_quantile : float
        Quantile threshold to define peak demand

    Returns
    -------
    dict
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error = y_pred - y_true

    # -------------------------------------------------
    # 1️⃣ Cost of Forecast Error
    # -------------------------------------------------
    over_error = np.clip(error, 0, None)
    under_error = np.clip(-error, 0, None)

    cost_over = over_error.sum() * unit_cost_over
    cost_under = under_error.sum() * unit_cost_under

    total_cost = cost_over + cost_under

    # -------------------------------------------------
    # 2️⃣ Peak Demand Miss Rate
    # -------------------------------------------------
    peak_threshold = np.quantile(y_true, peak_quantile)
    peak_mask = y_true >= peak_threshold

    if peak_mask.any():
        peak_miss_rate = (y_pred[peak_mask] < y_true[peak_mask]).mean()
    else:
        peak_miss_rate = None

    # -------------------------------------------------
    # 3️⃣ Over / Under Forecast Ratio
    # -------------------------------------------------
    over_ratio = (error > 0).mean()
    under_ratio = (error < 0).mean()

    # -------------------------------------------------
    # 4️⃣ Bias
    # -------------------------------------------------
    bias = error.mean()

    return {
        "cost_of_forecast_error": float(total_cost),
        "cost_overproduction": float(cost_over),
        "cost_underproduction": float(cost_under),
        "peak_demand_miss_rate": float(peak_miss_rate) if peak_miss_rate is not None else None,
        "over_forecast_ratio": float(over_ratio),
        "under_forecast_ratio": float(under_ratio),
        "bias": float(bias),
    }