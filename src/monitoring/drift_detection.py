import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def check_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float = 0.5,
    return_ratio: bool = False,
):
    """
    Calculates feature drift ratio.

    If return_ratio=True → returns drift_ratio (float)
    Else → returns boolean drift decision
    """

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    result = report.as_dict()

    drifted_features = 0
    total_features = 0

    for metric in result["metrics"]:
        if metric["metric"] == "DataDriftTable":
            total_features = metric["result"]["number_of_columns"]
            drifted_features = metric["result"]["number_of_drifted_columns"]

    if total_features == 0:
        drift_ratio = 0.0
    else:
        drift_ratio = drifted_features / total_features

    print(f"Drift ratio: {drift_ratio:.2f}")

    if return_ratio:
        return drift_ratio

    return drift_ratio > drift_threshold