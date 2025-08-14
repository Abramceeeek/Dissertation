import pandas as pd

def get_r_minus_q(sim_date, dividend_file, riskfree_file, maturity_days=365):
    df_div = pd.read_csv(dividend_file)
    df_div['date'] = pd.to_datetime(df_div['date'])
    
    df_rf = pd.read_csv(riskfree_file)
    df_rf['date'] = pd.to_datetime(df_rf['date'])
    
    div_row = df_div.iloc[(df_div['date'] - pd.Timestamp(sim_date)).abs().argsort()[:1]]
    q = div_row['rate'].values[0]

    rf_rows = df_rf.iloc[(df_rf['date'] - pd.Timestamp(sim_date)).abs().argsort()[:1]]
    rf_rows = rf_rows.copy()
    rf_rows['abs_maturity'] = (rf_rows['maturity_days'] - maturity_days).abs()
    rf_row = rf_rows.iloc[rf_rows['abs_maturity'].argsort()[:1]]
    r = rf_row['rate'].values[0]

    print(f"On {sim_date}: Risk-free rate ≈ {r:.4f}, Dividend yield ≈ {q:.4f}, (r - q) ≈ {(r - q):.4f}")

    return r - q


import pandas as pd
import numpy as np

def apply_rila_payoff(returns, buffer=0.1, cap=0.5, participation=1.0):
    returns = np.array(returns)
    credited_returns = np.zeros_like(returns)
    
    positive_mask = returns >= 0
    credited_returns[positive_mask] = np.minimum(returns[positive_mask], cap) * participation
    
    negative_mask = returns < 0
    loss_magnitude = np.abs(returns[negative_mask])
    
    within_buffer_mask = negative_mask & (np.abs(returns) <= buffer)
    credited_returns[within_buffer_mask] = 0
    
    beyond_buffer_mask = negative_mask & (np.abs(returns) > buffer)
    credited_returns[beyond_buffer_mask] = (returns[beyond_buffer_mask] + buffer) * participation
    
    return credited_returns

def apply_annual_rila_payoff(annual_returns, buffer=0.1, cap=0.12, participation=1.0, fee_rate=0.01):
    n_years, n_paths = annual_returns.shape
    account_values = np.ones(n_paths)
    
    for year in range(n_years):
        year_returns = annual_returns[year, :]
        
        credited_returns = apply_rila_payoff(year_returns, buffer, cap, participation)
        
        account_values *= (1 + credited_returns)
        
        account_values *= (1 - fee_rate)
    
    return account_values

def get_r_for_discounting(target_date, rf_path, maturity_days=7*365):
    rf_df = pd.read_csv(rf_path)
    rf_df['date'] = pd.to_datetime(rf_df['date'])
    target_date = pd.to_datetime(target_date)

    # Use 'maturity_days' instead of 'days'
    if 'maturity_days' not in rf_df.columns:
        raise KeyError("The column 'maturity_days' is missing in the interest rate data.")

    rf_rows = rf_df[rf_df['date'] == target_date]

    if rf_rows.empty:
        closest_date = rf_df['date'].iloc[(rf_df['date'] - target_date).abs().argsort().iloc[0]]
        print(f"[INFO] No data for {target_date.date()}, using closest available date: {closest_date.date()}")
        rf_rows = rf_df[rf_df['date'] == closest_date]

    rf_rows['abs_maturity'] = (rf_rows['maturity_days'] - maturity_days).abs()

    best_row = rf_rows.loc[rf_rows['abs_maturity'].idxmin()]
    return best_row['rate']
