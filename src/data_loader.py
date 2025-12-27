import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dropout"] = (df["Target"] == "Dropout").astype(int)
    return df
