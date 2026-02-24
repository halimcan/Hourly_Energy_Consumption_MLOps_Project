from prefect import flow
from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost
# from src.training.train_prophet import train_prophet  # ileride

import json
import joblib
from pathlib import Path
import polars as pl


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

    # ==========================================================
    # Yeni production config (temiz başlangıç)
    # ==========================================================
    production_config = {
        "mlflow": {
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "energy_forecasting",
            "run_id": None,
            "model_uri": None
        },
        "states": {}
    }

    # ==========================================================
    # LOOP – HER STATE
    # ==========================================================
    for file in parquet_files:

        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]
        print(f"State: {state}")

        # -------------------------
        # Train models
        # -------------------------
        results = []

        baseline_result = train_baseline(str(file), "target")
        results.append(baseline_result)

        xgb_result = train_xgboost(str(file), "target")
        results.append(xgb_result)

        # prophet_result = train_prophet(...)
        # results.append(prophet_result)

        for r in results:
            print(f"{r['model_name']} MAE: {r['metrics']['mae']}")

        # -------------------------
        # Compare (MODEL-AGNOSTIC)
        # -------------------------
        winner = min(
            results,
            key=lambda x: x["metrics"]["mae"]
        )

        winner_name = winner["model_name"]
        winner_model = winner["model"]
        winner_metrics = winner["metrics"]
        winner_features = winner.get("feature_columns")

        print(f"🏆 Winner: {winner_name}")

        # -------------------------
        # Save artifacts (state bazlı)
        # -------------------------
        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        model_path = state_dir / "best_model.pkl"
        metadata_path = state_dir / "best_model.json"

        joblib.dump(winner_model, model_path)

        metadata = {
            "state": state,
            "model_type": winner_name,
            "metrics": winner_metrics,
            "data_file": str(file)
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"📦 Model saved to {state_dir}")

        # -------------------------
        # production.json → states
        # -------------------------
        production_config["states"][state] = {
            "model_type": winner_name,
            "loader": "local",
            "model_path": str(model_path),
            "feature_names": winner_features  # ✅ HARDCODE YOK
        }

    # ==========================================================
    # production.json overwrite
    # ==========================================================
    production_file.write_text(
        json.dumps(production_config, indent=2),
        encoding="utf-8"
    )

    print("\n📝 production.json state-aware olarak güncellendi")
    print("✅ Tüm state'ler için promotion tamamlandı")

    return {
        "states_processed": list(production_config["states"].keys())
    }


if __name__ == "__main__":
    evaluate_and_promote()