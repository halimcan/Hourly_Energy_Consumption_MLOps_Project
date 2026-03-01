import polars as pl
from src.monitoring.simulation_runner import run_simulation

# 2014–2018 tam dataset'i yükle
df = pl.read_parquet("data/processed/DAYTON_hourly_processed.parquet")

run_simulation(df)