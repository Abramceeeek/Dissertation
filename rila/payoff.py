"""
RILA payoff logic for single-term and annual-reset products, with buffer, cap, fee, and participation options.
"""
import numpy as np

def apply_rila_payoff(returns, buffer, cap):
    """
    Vectorized RILA payoff logic for single-term contracts.
    Args:
        returns: np.array of raw returns (e.g., (S_T - S_0)/S_0)
        buffer: downside buffer (e.g., 0.1 for 10%)
        cap: upside cap (e.g., 0.5 for 50%)
    Returns:
        np.array of credited returns (same shape as input)
    """
    credited = np.where(
        returns >= 0,
        np.minimum(returns, cap),
        np.where(np.abs(returns) <= buffer, 0, returns + buffer)
    )
    return credited

def apply_rila_annual_reset(account_start, annual_returns, buffer, cap, fee=0.0, participation=1.0):
    """
    Apply RILA logic with annual reset, buffer/cap, and optional fee.
    Args:
        account_start: initial account value (scalar)
        annual_returns: np.ndarray shape (n_years, n_paths)
        buffer: downside buffer (e.g., 0.1)
        cap: upside cap (e.g., 0.12)
        fee: annual fee (e.g., 0.01 for 1%)
        participation: participation rate (default 1.0)
    Returns:
        np.ndarray of final account values (n_paths,)
    """
    n_years, n_paths = annual_returns.shape
    account = np.full(n_paths, account_start, dtype=np.float64)
    for year in range(n_years):
        capped = np.minimum(annual_returns[year], cap)
        buffered = np.where(capped >= -buffer, capped, capped + buffer)
        credited = np.where(capped >= -buffer, capped, buffered)
        account *= (1 + participation * credited)
        account *= (1 - fee)
    return account 