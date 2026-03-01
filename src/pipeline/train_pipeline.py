from prefect import flow
import subprocess
import sys


def run_step(cmd: list[str], step_name: str):
    print(f"\n========== {step_name} ==========")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ {step_name} FAILED")
        sys.exit(1)

    print(f"✅ {step_name} COMPLETED")


@flow(name="training-pipeline-flow")
def train_pipeline():

    run_step(
        ["python", "-m", "src.pipeline.data_preparation"],
        "Feature Engineering",
    )

    run_step(
        ["python", "-m", "src.training.evaluate_and_promote_flow"],
        "Train & Evaluate Models",
    )

    run_step(
        ["python", "-m", "src.registry.promote_best"],
        "Promote Model to Registry",
    )

    print("\n🚀 FULL PIPELINE SUCCESSFULLY COMPLETED")


if __name__ == "__main__":
    train_pipeline()