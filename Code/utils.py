import pandas as pd

def get_r_minus_q(sim_date, dividend_file, riskfree_file, maturity_days=365):
    """
    Returns the risk-neutral drift adjustment (r - q) for a given simulation date and term.
    """
    # Load dividend yield data
    df_div = pd.read_csv(dividend_file)
    df_div['date'] = pd.to_datetime(df_div['date'])
    
    # Load risk-free curve data
    df_rf = pd.read_csv(riskfree_file)
    df_rf['date'] = pd.to_datetime(df_rf['date'])
    
    # Find nearest date in dividend data
    div_row = df_div.iloc[(df_div['date'] - pd.Timestamp(sim_date)).abs().argsort()[:1]]
    q = div_row['rate'].values[0]

    # Find nearest date in risk-free data
    rf_rows = df_rf.iloc[(df_rf['date'] - pd.Timestamp(sim_date)).abs().argsort()[:1]]
    # Choose the maturity closest to desired (e.g. 1-year)
    rf_rows = rf_rows.copy()
    rf_rows['abs_maturity'] = (rf_rows['maturity_days'] - maturity_days).abs()
    rf_row = rf_rows.iloc[rf_rows['abs_maturity'].argsort()[:1]]
    r = rf_row['rate'].values[0]

    print(f"✅ On {sim_date}: Risk-free rate ≈ {r:.4f}, Dividend yield ≈ {q:.4f}, (r - q) ≈ {(r - q):.4f}")

    return r - q


def get_r_for_discounting(sim_date, riskfree_file, maturity_days=365):
    """
    Returns risk-free rate for a given simulation date and term (for discounting).
    """
    df_rf = pd.read_csv(riskfree_file)
    df_rf['date'] = pd.to_datetime(df_rf['date'])
    
    # Find nearest date
    rf_rows = df_rf.iloc[(df_rf['date'] - pd.Timestamp(sim_date)).abs().argsort()[:1]]
    rf_rows = rf_rows.copy()
    rf_rows['abs_maturity'] = (rf_rows['maturity_days'] - maturity_days).abs()
    rf_row = rf_rows.iloc[rf_rows['abs_maturity'].argsort()[:1]]
    r = rf_row['rate'].values[0]
    
    print(f"✅ For {sim_date}: Risk-free rate for discounting ≈ {r:.4f}")
    return r
