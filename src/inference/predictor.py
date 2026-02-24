# src/inference/predictor.py

def predict(model, model_type: str, data):
    """
    Unified inference entrypoint.
    """

    model_type = model_type.lower()

    if model_type == "baseline":
        # data = geçmiş dataframe
        return model.forecast_next_24(data)

    elif model_type == "xgboost":
        # data = feature matrix
        return model.predict(data)

    elif model_type == "prophet":
        # data = prophet-style dataframe
        return model.predict(data)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")