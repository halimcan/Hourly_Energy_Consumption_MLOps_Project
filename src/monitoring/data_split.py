import polars as pl


def split_train_production(df: pl.DataFrame):
    """
    2014-2017 → training
    2018 → production simulation
    """

    train = df.filter(pl.col("year") <= 2017)
    production = df.filter(pl.col("year") == 2018)

    if train.is_empty():
        raise ValueError("Training dataset boş.")

    if production.is_empty():
        raise ValueError("Production dataset boş (2018 bulunamadı).")

    return train, production