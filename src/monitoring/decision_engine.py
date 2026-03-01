from src.metrics.business_metrics import calculate_business_metrics


def should_retrain(
    drift_flag: bool,
    y_true_model,
    y_pred_model,
    y_true_baseline,
    y_pred_baseline,
    tolerance: float = 0.10,
) -> bool:
    """
    Advanced decision logic:

    Retrain if:
    1. Drift detected
    2. Model worse than baseline (RMSE tolerance)
    3. Business cost higher than baseline
    """

    import numpy as np

    # ----------------------------
    # Technical Metric (RMSE)
    # ----------------------------
    model_rmse = np.sqrt(np.mean((y_true_model - y_pred_model) ** 2))
    baseline_rmse = np.sqrt(np.mean((y_true_baseline - y_pred_baseline) ** 2))

    performance_bad = model_rmse > baseline_rmse * (1 + tolerance)

    # ----------------------------
    # Business Metrics
    # ----------------------------
    model_business = calculate_business_metrics(
        y_true_model, y_pred_model
    )

    baseline_business = calculate_business_metrics(
        y_true_baseline, y_pred_baseline
    )

    business_bad = (
        model_business["cost_of_forecast_error"]
        > baseline_business["cost_of_forecast_error"]
    )

    # ----------------------------
    # Final Decision
    # ----------------------------
    if drift_flag and performance_bad and business_bad:
        return True

    return False