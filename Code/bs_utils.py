import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price_call(S, K, T, r, q, sigma):
    """
    Black-Scholes price for a European call.
    """
    if T <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def bs_implied_vol(price, S, K, T, r, q, tol=1e-6):
    """
    Numerical inversion of BS formula to get implied vol.
    """
    if price < max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) + 1e-5:
        return 0.0001  # Near-zero vol

    def objective(sigma):
        return bs_price_call(S, K, T, r, q, sigma) - price

    try:
        iv = brentq(objective, 1e-5, 5.0, xtol=tol)
    except ValueError:
        iv = np.nan
    return iv

if __name__ == "__main__":
    # Quick test
    S = 4500
    K = 4500
    T = 1.0
    r = 0.02
    q = 0.01
    sigma = 0.25

    price = bs_price_call(S, K, T, r, q, sigma)
    print(f"✅ BS price: {price:.2f}")

    iv = bs_implied_vol(price, S, K, T, r, q)
    print(f"✅ Inverted IV: {iv:.4f}")
