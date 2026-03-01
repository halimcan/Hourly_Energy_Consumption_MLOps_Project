from prefect import flow
from pathlib import Path
import json
import joblib
import polars as pl
import numpy as np
import mlflow

from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost
from src.training.train_prophet import train_prophet
from src.metrics.business_metrics import calculate_business_metrics


@flow(name="evaluate-and-promote-multi-state")
def evaluate_and_promote():

    processed_dir = Path("data/processed")
    artifact_root = Path("src/registry/artifacts")
    production_file = Path("src/registry/production.json")

    parquet_files = list(processed_dir.glob("*_processed.parquet"))

    if not parquet_files:
        print("❌ Processed veri bulunamadı.")
        return

    print(f"\n🚀 Toplam {len(parquet_files)} state işlenecek.\n")

    production_config = {
        "mlflow": {
            "tracking_uri": "file:./mlruns",
            "experiment_name": "energy_forecasting"
        },
        "states": {}
    }

    for file in parquet_files:

        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]

        print(f"State: {state}")

        baseline_result = train_baseline(str(file), "target")
        xgb_result = train_xgboost(str(file), "target")
        prophet_result = train_prophet(str(file), "target")

        results = {
            "baseline": baseline_result,
            "xgboost": xgb_result,
            "prophet": prophet_result
        }

        winner_name = min(
            results,
            key=lambda k: results[k]["metrics"]["rmse"]
        )

        winner_result = results[winner_name]
        winner_model = winner_result["model"]
        winner_metrics = winner_result["metrics"]

        print(f"\n🏆 Winner Model: {winner_name}")

        # MLflow run_id yakala
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None

        # Business metrics
        df_pd = df.to_pandas().sort_values("datetime")
        y_true = df_pd["target"].values[-24:]

        if winner_name == "baseline":
            y_pred = winner_model.forecast_next_24(df_pd.iloc[:-24])

        elif winner_name == "xgboost":
            features = df_pd.iloc[-24:].drop(columns=["target"])
            y_pred = winner_model.predict(features)

        elif winner_name == "prophet":
            future = winner_model.make_future_dataframe(
                periods=24,
                freq="H"
            )
            forecast = winner_model.predict(future)
            y_pred = forecast.tail(24)["yhat"].values

        y_pred = np.array(y_pred)

        business_metrics = calculate_business_metrics(
            y_true=y_true,
            y_pred=y_pred
        )

        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        model_path = state_dir / "best_model.pkl"
        metadata_path = state_dir / "best_model.json"

        joblib.dump(winner_model, model_path)

        feature_names = [] if winner_name == "prophet" else [
            "hour", "dayofweek", "month", "year", "is_weekend",
            "sin_hour", "cos_hour",
            "target_lag_1", "target_lag_24",
            "target_roll_mean_24", "target_roll_std_24"
        ]

        metadata = {
            "state": state,
            "model_type": winner_name,
            "run_id": run_id,
            "feature_names": feature_names,
            "rmse": winner_metrics.get("rmse"),
            "cost_of_forecast_error": business_metrics.get("cost_of_forecast_error"),
            "business_metrics": business_metrics,
            "data_file": str(file)
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"📦 Model saved → {state_dir}")

        production_config["states"][state] = {
            "model_type": winner_name,
            "loader": "local",
            "model_path": str(model_path),
            "feature_names": feature_names,
            "run_id": run_id
        }

    production_file.write_text(
        json.dumps(production_config, indent=2),
        encoding="utf-8"
    )

    print("\n📝 production.json güncellendi")
    print("✅ Tüm state'ler için promotion tamamlandı")

    return {
        "states_processed": list(production_config["states"].keys())
    }


if __name__ == "__main__":
    evaluate_and_promote()