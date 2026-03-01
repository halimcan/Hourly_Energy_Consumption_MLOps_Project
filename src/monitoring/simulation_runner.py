import polars as pl
import numpy as np
import joblib
from pathlib import Path

from src.monitoring.decision_engine import should_retrain
from src.monitoring.drift_detection import check_data_drift
from src.monitoring.data_split import split_train_production
from src.metrics.business_metrics import calculate_business_metrics


ROLLING_WINDOW_DAYS = 7
DRIFT_CHECK_DAYS = 7
TOLERANCE = 0.10

FEATURE_COLS = [
    "hour", "dayofweek", "month", "year", "is_weekend",
    "sin_hour", "cos_hour",
    "target_lag_1", "target_lag_24",
    "target_roll_mean_24", "target_roll_std_24"
]


def load_production_model(state: str):
    model_path = Path(f"src/registry/artifacts/{state}/best_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    return joblib.load(model_path)


def compute_persistence_predictions(df: pl.DataFrame):
    df = df.sort("Datetime").to_pandas()

    y_true = df["target"].values[24:]
    y_pred = df["target"].values[:-24]

    return y_true, y_pred


def compute_model_predictions(model, df: pl.DataFrame):
    df = df.sort("Datetime").to_pandas()

    X = df[FEATURE_COLS]
    y_true = df["target"].values
    y_pred = model.predict(X)

    return y_true, y_pred


def run_simulation(df: pl.DataFrame, state: str) -> dict:

    train, production = split_train_production(df)

    model = load_production_model(state)

    y_true_base, y_pred_base = compute_persistence_predictions(production)

    production_sorted = production.sort("Datetime")
    production_aligned = production_sorted[24:]

    y_true_model, y_pred_model = compute_model_predictions(
        model, production_aligned
    )

    window_hours = ROLLING_WINDOW_DAYS * 24
    drift_check_interval = DRIFT_CHECK_DAYS * 24

    train_pd = train.to_pandas()
    production_pd = production_aligned.to_pandas()

    last_metrics = None

    for i in range(window_hours, len(y_true_model)):

        if i % drift_check_interval != 0:
            continue

        y_t_model = y_true_model[i - window_hours:i]
        y_p_model = y_pred_model[i - window_hours:i]

        y_t_base = y_true_base[i - window_hours:i]
        y_p_base = y_pred_base[i - window_hours:i]

        model_rmse = float(np.sqrt(np.mean((y_t_model - y_p_model) ** 2)))
        baseline_rmse = float(np.sqrt(np.mean((y_t_base - y_p_base) ** 2)))

        current_window_df = production_pd.iloc[i - window_hours:i]

        drift_ratio = check_data_drift(
            reference_df=train_pd[FEATURE_COLS],
            current_df=current_window_df[FEATURE_COLS],
            drift_threshold=0.5,
            return_ratio=True
        )

        drift_flag = drift_ratio > 0.5

        model_business = calculate_business_metrics(
            y_t_model, y_p_model
        )

        baseline_business = calculate_business_metrics(
            y_t_base, y_p_base
        )

        retrain_flag = should_retrain(
            drift_flag=drift_flag,
            y_true_model=y_t_model,
            y_pred_model=y_p_model,
            y_true_baseline=y_t_base,
            y_pred_baseline=y_p_base,
            tolerance=TOLERANCE
        )

        last_metrics = {
            "state": state,
            "drift_ratio": drift_ratio,
            "model_rmse": model_rmse,
            "baseline_rmse": baseline_rmse,
            "business_cost_model": model_business["cost_of_forecast_error"],
            "business_cost_baseline": baseline_business["cost_of_forecast_error"],
            "retrain_triggered": retrain_flag,
        }

    return last_metrics


if __name__ == "__main__":
    test_state = "DAYTON"
    df = pl.read_parquet(
        f"data/processed/{test_state}_hourly_processed.parquet"
    )
    print(run_simulation(df, test_state))