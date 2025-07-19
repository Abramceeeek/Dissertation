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