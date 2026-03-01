from prefect import flow
from src.monitoring.multi_state_monitor import run_multi_state_monitoring
from src.pipeline.train_pipeline import train_pipeline


@flow(name="multi-state-monitoring-flow")
def monitoring_flow():

    retrain_any = run_multi_state_monitoring()

    # Eğer en az bir state retrain isterse global retrain çalıştır
    if retrain_any:
        print("🔥 At least one state requires retraining.")
        train_pipeline()
    else:
        print("✅ No retraining required.")


if __name__ == "__main__":
    monitoring_flow()