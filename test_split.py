import polars as pl
from src.monitoring.simulation_runner import run_simulation

df = pl.read_parquet("data/processed/DAYTON_hourly_processed.parquet")

run_simulation(df)