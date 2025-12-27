import pandas as pd
from .config import SATISFACTION_APPROVAL_THRESHOLD

def add_academic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["approval_ratio"] = (
        df["Curricular units 1st sem (approved)"] /
        df["Curricular units 1st sem (enrolled)"]
    ).fillna(0)

    df["evaluation_ratio"] = (
        df["Curricular units 1st sem (evaluations)"] /
        df["Curricular units 1st sem (enrolled)"]
    ).fillna(0)

    return df

def add_satisfaction_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["satisfied_proxy"] = (
        (df["dropout"] == 0) &
        (df["approval_ratio"] >= SATISFACTION_APPROVAL_THRESHOLD)
    ).astype(int)

    return df
