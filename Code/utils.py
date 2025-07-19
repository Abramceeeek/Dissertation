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


import pandas as pd
import numpy as np

def apply_rila_payoff(returns, buffer=0.1, cap=0.5, participation=1.0):
    """
    Apply RILA (Registered Index-Linked Annuity) payoff logic to returns.
    
    Parameters:
    - returns: array of raw returns (S_T/S_0 - 1)
    - buffer: downside buffer level (e.g., 0.1 for 10% buffer)
    - cap: upside cap level (e.g., 0.5 for 50% cap)
    - participation: participation rate (typically 1.0)
    
    Returns:
    - credited_returns: array of credited returns after applying buffer and cap
    """
    returns = np.array(returns)
    credited_returns = np.zeros_like(returns)
    
    # Positive returns: apply cap
    positive_mask = returns >= 0
    credited_returns[positive_mask] = np.minimum(returns[positive_mask], cap) * participation
    
    # Negative returns: apply buffer
    negative_mask = returns < 0
    loss_magnitude = np.abs(returns[negative_mask])
    
    # Within buffer: no loss
    within_buffer_mask = negative_mask & (np.abs(returns) <= buffer)
    credited_returns[within_buffer_mask] = 0
    
    # Beyond buffer: loss minus buffer amount
    beyond_buffer_mask = negative_mask & (np.abs(returns) > buffer)
    credited_returns[beyond_buffer_mask] = (returns[beyond_buffer_mask] + buffer) * participation
    
    return credited_returns

def apply_annual_rila_payoff(annual_returns, buffer=0.1, cap=0.12, participation=1.0, fee_rate=0.01):
    """
    Apply annual RILA logic with yearly resets and fees.
    
    Parameters:
    - annual_returns: 2D array of shape (n_years, n_paths) with yearly returns
    - buffer: annual downside buffer
    - cap: annual upside cap
    - participation: participation rate
    - fee_rate: annual fee rate
    
    Returns:
    - final_account_values: final account values after all years
    """
    n_years, n_paths = annual_returns.shape
    account_values = np.ones(n_paths)  # Start with $1
    
    for year in range(n_years):
        year_returns = annual_returns[year, :]
        
        # Apply RILA logic for this year
        credited_returns = apply_rila_payoff(year_returns, buffer, cap, participation)
        
        # Update account values
        account_values *= (1 + credited_returns)
        
        # Apply annual fee
        account_values *= (1 - fee_rate)
    
    return account_values

def get_r_for_discounting(target_date, rf_path, maturity_days=7*365):
    rf_df = pd.read_csv(rf_path)
    rf_df['date'] = pd.to_datetime(rf_df['date'])
    target_date = pd.to_datetime(target_date)

    # Check if the days column exists
    if 'days' not in rf_df.columns:
        raise KeyError("The column 'days' is missing in the interest rate data.")

    # Find the rows for the given date
    rf_rows = rf_df[rf_df['date'] == target_date]

    if rf_rows.empty:
        # fallback: find the closest date instead
        closest_date = rf_df['date'].iloc[(rf_df['date'] - target_date).abs().argsort().iloc[0]]
        print(f"[INFO] No data for {target_date.date()}, using closest available date: {closest_date.date()}")
        rf_rows = rf_df[rf_df['date'] == closest_date]

    # Calculate absolute difference to find nearest maturity
    rf_rows['abs_maturity'] = (rf_rows['days'] - maturity_days).abs()

    best_row = rf_rows.loc[rf_rows['abs_maturity'].idxmin()]
    return best_row['rate']
