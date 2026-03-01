import streamlit as st
import requests
import polars as pl
import pandas as pd
import plotly.express as px
import mlflow
from pathlib import Path
from datetime import datetime

API_URL = "http://ml_ops_app:8000"

st.set_page_config(page_title="Energy MLOps Platform", layout="wide")

st.title("⚡ Energy Forecasting MLOps Platform")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediction", "📊 Monitoring", "📈 Model Registry"]
)

# ===================================================
# 🔮 PREDICTION
# ===================================================
with tab1:

    st.subheader("Energy Consumption Forecast")

    states = [
        "DAYTON","NI","EKPC","DOM","COMED","DEOK",
        "FE","AEP","DUQ","PJMW","PJM","PJME"
    ]

    selected_state = st.selectbox("Select State", states)

    selected_date = st.date_input("Select Date")
    selected_time = st.time_input("Select Time")

    if st.button("Predict"):

        selected_datetime = datetime.combine(selected_date, selected_time)

        payload = {
            "state": selected_state,
            "datetime": selected_datetime.isoformat()
        }

        try:
            response = requests.post(
                f"{API_URL}/predict-from-datetime",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Prediction: {round(result['prediction'], 2)} MW"
                )
            else:
                st.error(response.json())

        except Exception as e:
            st.error(f"API Error: {str(e)}")


# ===================================================
# 📊 MONITORING
# ===================================================
with tab2:

    st.subheader("Monitoring Dashboard")

    LOG_PATH = Path("logs/monitoring_results.parquet")

    if not LOG_PATH.exists():
        st.warning("No monitoring data found.")
    else:

        df = pl.read_parquet(LOG_PATH).to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        states = df["state"].unique()
        selected_state = st.selectbox(
            "Select State",
            states,
            key="monitor_state"
        )

        state_df = df[df["state"] == selected_state]

        st.markdown("### Drift Ratio")
        fig_drift = px.line(
            state_df,
            x="timestamp",
            y="drift_ratio",
            markers=True,
        )
        fig_drift.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_drift, use_container_width=True)

        st.markdown("### RMSE Comparison")
        fig_rmse = px.line(
            state_df,
            x="timestamp",
            y=["model_rmse", "baseline_rmse"],
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown("### Business Cost")
        fig_cost = px.line(
            state_df,
            x="timestamp",
            y=["business_cost_model", "business_cost_baseline"],
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("### Retrain Events")
        retrain_df = state_df[state_df["retrain_triggered"] == True]

        if retrain_df.empty:
            st.success("No retraining events detected.")
        else:
            st.error("Retraining triggered at:")
            st.dataframe(
                retrain_df[
                    ["timestamp", "drift_ratio", "model_rmse"]
                ]
            )


# ===================================================
# 📈 MODEL REGISTRY + EXPERIMENTS
# ===================================================
with tab3:

    st.subheader("MLflow Model Registry & Experiments")

    mlflow.set_tracking_uri("file:./mlruns")

    experiments = mlflow.search_experiments()

    if not experiments:
        st.warning("No MLflow experiments found.")
    else:

        exp_names = [exp.name for exp in experiments]
        selected_exp = st.selectbox(
            "Select Experiment",
            exp_names
        )

        exp = mlflow.get_experiment_by_name(selected_exp)

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id]
        )

        if runs.empty:
            st.warning("No runs in this experiment.")
        else:

            st.markdown("### Experiment Runs")

            display_cols = [
                col for col in runs.columns
                if col.startswith("metrics.") or col.startswith("params.")
            ]

            display_cols = ["start_time"] + display_cols

            st.dataframe(
                runs[display_cols].sort_values(
                    "start_time",
                    ascending=False
                )
            )

    # --------------------------------------------------
# Production Registry Overview + MLflow Match
# --------------------------------------------------

st.markdown("### Active Production Models")

try:
    from src.api.model_loader import load_registry

    registry = load_registry()
    states_block = registry.get("states", {})

    if states_block:

        rows = []

        for state, cfg in states_block.items():

            run_id = cfg.get("run_id")
            rmse = None
            cost = None

            if run_id:
                try:
                    run_info = mlflow.get_run(run_id)
                    rmse = run_info.data.metrics.get("rmse")
                    cost = run_info.data.metrics.get("cost_of_forecast_error")
                except:
                    pass

            rows.append({
                "state": state,
                "model_type": cfg.get("model_type"),
                "run_id": run_id,
                "rmse": rmse,
                "business_cost": cost,
                "artifact_path": cfg.get("artifact_path"),
            })

        prod_df = pd.DataFrame(rows)
        st.dataframe(prod_df)

    else:
        st.info("No production models registered.")

except Exception as e:
    st.error(f"Registry read error: {str(e)}")