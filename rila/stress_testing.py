import pandas as pd
import numpy as np

def covid_stress_test(paths: pd.DataFrame, start_date: str = '2020-02-19', shock: float = -0.3) -> pd.DataFrame:
    idx = paths.index.get_loc(pd.Timestamp(start_date))
    shocked = paths.copy()
    shocked.iloc[idx:] *= (1 + shock)
    return shocked 