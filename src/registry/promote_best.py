from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_ROOT = ROOT / "src" / "registry" / "artifacts"
PRODUCTION_FILE = ROOT / "src" / "registry" / "production.json"


def promote(state: str):

    state = state.upper()

    state_dir = ARTIFACT_ROOT / state
    metadata_path = state_dir / "best_model.json"
    model_path = state_dir / "best_model.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"{state} için metadata bulunamadı.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type")
    feature_names = metadata.get("feature_names", [])
    run_id = metadata.get("run_id")

    # production.json varsa oku
    if PRODUCTION_FILE.exists():
        with open(PRODUCTION_FILE, "r") as f:
            production_config = json.load(f)
    else:
        production_config = {
            "mlflow": {
                "tracking_uri": "file:./mlruns",
                "experiment_name": "energy_forecasting"
            },
            "states": {}
        }

    production_config["states"][state] = {
        "model_type": model_type,
        "loader": "local",
        "model_path": str(model_path),
        "feature_names": feature_names,
        "run_id": run_id
    }

    PRODUCTION_FILE.write_text(
        json.dumps(production_config, indent=2),
        encoding="utf-8"
    )

    print(f"🚀 {state} production'a promote edildi.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Kullanım: python -m src.registry.promote_best STATE")
        sys.exit(1)

    promote(sys.argv[1])