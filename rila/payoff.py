import numpy as np

def apply_rila_payoff(returns, buffer, cap):
    credited = np.where(
        returns >= 0,
        np.minimum(returns, cap),
        np.where(np.abs(returns) <= buffer, 0, returns + buffer)
    )
    return credited

def apply_rila_annual_reset(account_start, annual_returns, buffer, cap, fee=0.0, participation=1.0):
    n_years, n_paths = annual_returns.shape
    account = np.full(n_paths, account_start, dtype=np.float64)
    for year in range(n_years):
        capped = np.minimum(annual_returns[year], cap)
        buffered = np.where(capped >= -buffer, capped, capped + buffer)
        credited = np.where(capped >= -buffer, capped, buffered)
        account *= (1 + participation * credited)
        account *= (1 - fee)
    return account 