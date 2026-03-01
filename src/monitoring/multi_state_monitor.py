import polars as pl
from pathlib import Path
from datetime import datetime

from src.api.model_loader import load_registry
from src.monitoring.simulation_runner import run_simulation


MONITORING_LOG = Path("logs/monitoring_results.parquet")
MONITORING_LOG.parent.mkdir(parents=True, exist_ok=True)


def get_all_states():
    registry = load_registry()
    states_block = registry.get("states") or registry.get("state_models") or {}

    if not isinstance(states_block, dict):
        return []

    return list(states_block.keys())


def log_monitoring_result(metrics: dict):

    if metrics is None:
        return

    metrics["timestamp"] = datetime.utcnow()

    df_new = pl.DataFrame([metrics])

    if MONITORING_LOG.exists():
        df_old = pl.read_parquet(MONITORING_LOG)
        df_all = pl.concat([df_old, df_new])
    else:
        df_all = df_new

    df_all.write_parquet(MONITORING_LOG)


def run_multi_state_monitoring() -> bool:

    states = get_all_states()

    if not states:
        print("No states found in registry.")
        return False

    print("=======================================")
    print("Running monitoring for all states")
    print("=======================================")

    any_retrain = False

    for state in states:

        print(f"\nMonitoring state: {state}")

        data_path = Path(
            f"data/processed/{state}_hourly_processed.parquet"
        )

        if not data_path.exists():
            print(f"⚠️ Data not found for state: {state}")
            continue

        df = pl.read_parquet(data_path)

        metrics = run_simulation(df, state)

        log_monitoring_result(metrics)

        if metrics and metrics.get("retrain_triggered"):
            any_retrain = True

        print(f"{state} | Retrain needed: {metrics.get('retrain_triggered')}")

    print("\nMonitoring completed.")
    print(f"Global retrain required: {any_retrain}")

    return any_retrain