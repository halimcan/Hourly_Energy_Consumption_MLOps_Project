def train_xgboost(data_path, target_col="target"):

    df = pl.read_parquet(data_path).to_pandas()

    features = [
        "hour", "dayofweek", "month", "year", "is_weekend",
        "sin_hour", "cos_hour",
        "target_lag_1", "target_lag_24",
        "target_roll_mean_24", "target_roll_std_24"
    ]

    existing_features = [f for f in features if f in df.columns]

    # --- TRAIN / TEST SPLIT ---
    train_df = df[df["year"] <= 2017]
    test_df = df[df["year"] == 2018]

    X_train = train_df[existing_features]
    y_train = train_df[target_col]

    X_test = test_df[existing_features]
    y_test = test_df[target_col]

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return {
        "model_name": "xgboost",
        "model": model,
        "metrics": {"mae": float(mae)},
        "mae": float(mae),
        "feature_columns": existing_features
    }