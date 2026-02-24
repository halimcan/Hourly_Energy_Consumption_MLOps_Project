import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import json
from pathlib import Path
from prefect import flow
from src.training.train_baseline import train_baseline

# 1. Evaluate_and_promote_flow.py'nin beklediği fonksiyonu ekliyoruz
def train_xgboost(data_path, target_col="target"):
    """
    XGBoost modelini eğitir ve sonuçları döner.
    """
    df = pl.read_parquet(data_path).to_pandas()
    
    # Feature listesi
    features = [
        "hour", "dayofweek", "month", "year", "is_weekend",
        "sin_hour", "cos_hour",
        "target_lag_1", "target_lag_24",
        "target_roll_mean_24", "target_roll_std_24"
    ]
    
    existing_features = [f for f in features if f in df.columns]
    X = df[existing_features]
    y = df[target_col]
    
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    
    return {
        "model_name": "xgboost",
        "model": model,
        "metrics": {"mae": float(mae)},
        "mae": float(mae),
        "feature_columns": existing_features
    }

# 2. Senin mevcut flow yapın (Hata almaması için dokunmuyoruz)
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
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "energy_forecasting",
            "run_id": None,
            "model_uri": None
        },
        "states": {}
    }

    for file in parquet_files:
        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]
        print(f"State: {state}")

        results = []
        
        # Baseline
        baseline_result = train_baseline(str(file), "target")
        results.append(baseline_result)

        # XGBoost (Artık yukarıda tanımlı olduğu için hata vermeyecek)
        xgb_result = train_xgboost(str(file), "target")
        results.append(xgb_result)

        for r in results:
            print(f"{r['model_name']} MAE: {r['metrics']['mae']}")

        winner = min(results, key=lambda x: x["metrics"]["mae"])
        winner_name = winner["model_name"]
        winner_model = winner["model"]
        winner_metrics = winner["metrics"]
        winner_features = winner.get("feature_columns")

        print(f"🏆 Winner: {winner_name}")

        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(winner_model, state_dir / "best_model.pkl")

        metadata = {
            "state": state,
            "model_type": winner_name,
            "metrics": winner_metrics,
            "data_file": str(file)
        }

        (state_dir / "best_model.json").write_text(json.dumps(metadata, indent=2))

        production_config["states"][state] = {
            "model_type": winner_name,
            "loader": "local",
            "model_path": str(state_dir / "best_model.pkl"),
            "feature_names": winner_features
        }

    production_file.write_text(json.dumps(production_config, indent=2))
    print("\n✅ Tüm süreç tamamlandı.")

if __name__ == "__main__":
    evaluate_and_promote()