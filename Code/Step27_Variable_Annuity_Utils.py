import numpy as np

def apply_rila_payoff(returns, buffer, cap):
    credited = np.where(
        returns >= 0,
        np.minimum(returns, cap),
        np.where(np.abs(returns) <= buffer, 0, returns + buffer)
    )
    return credited