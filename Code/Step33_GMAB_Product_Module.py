"""
GMAB (Guaranteed Minimum Accumulation Benefit) Product Module

This module implements the GMAB variable annuity product for the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
Purpose: GMAB product implementation with pricing and hedging functionality
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code'))

from Step22_Black_Scholes_Utils import bs_price_call
from Step23_Heston_Carr_Madan_Pricing import carr_madan_call_price

@dataclass
class GMABParams:
    """
    GMAB Product Parameters
    
    T_years: Contract maturity in years (default 7)
    g_annual: Guaranteed minimum accumulation rate (3% p.a.)
    fee_annual: Annual management fee
    trans_cost_bps: Transaction cost in basis points
    rebalance_freq: Rebalancing frequency in days (default weekly = 7 days)
    """
    T_years: float = 7.0
    g_annual: float = 0.03  # 3% p.a. guarantee
    fee_annual: float = 0.015  # 1.5% annual management fee
    trans_cost_bps: float = 10.0  # 10 basis points transaction cost
    rebalance_freq: int = 7  # Weekly rebalancing

def evolve_account_from_prices(S: np.ndarray, fee_annual: float, dt_years: float) -> np.ndarray:
    """
    Evolve account value from stock prices with continuous fee discount.
    
    Args:
        S: Stock price array of shape (n_steps+1, n_paths)
        fee_annual: Annual fee rate
        dt_years: Time step size in years
    
    Returns:
        A_t: Account value array of shape (n_steps+1, n_paths)
    """
    n_steps, n_paths = S.shape[0] - 1, S.shape[1]
    A = np.zeros_like(S)
    A[0] = S[0]  # Initial account value equals initial stock price (normalized)
    
    # Apply continuous fee discount
    fee_per_step = np.exp(-fee_annual * dt_years)
    
    for t in range(1, n_steps + 1):
        # Account grows with stock performance, discounted by fees
        A[t] = A[t-1] * (S[t] / S[t-1]) * fee_per_step
    
    return A

def gmab_value_and_delta(
    S: float, 
    A: float, 
    T: float, 
    r: float, 
    q: float, 
    gmab_params: GMABParams,
    model: str = 'bs',
    heston_params: dict = None,
    lambda_scale: float = 1.0,
    bump_size: float = 0.01
) -> Tuple[float, float]:
    """
    Compute GMAB value and delta using specified pricing model.
    
    The GMAB payoff at maturity is max(G - A_T, 0) where G is the guaranteed amount.
    This is equivalent to a put option on the account value with strike G.
    
    Args:
        S: Current stock price
        A: Current account value
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        gmab_params: GMAB product parameters
        model: Pricing model ('bs' or 'heston')
        heston_params: Heston model parameters (if model='heston')
        lambda_scale: Scaling factor for option value
        bump_size: Bump size for delta calculation (1% default)
    
    Returns:
        (value, delta): GMAB value and delta
    """
    if T <= 0:
        # At maturity: payoff = max(G - A_T, 0)
        G = A * (1 + gmab_params.g_annual) ** gmab_params.T_years  # Guaranteed amount
        payoff = max(G - A, 0.0)
        return payoff, 0.0
    
    # Calculate effective strike: G/A_t ratio gives us the strike for put on normalized account
    G = A * (1 + gmab_params.g_annual) ** gmab_params.T_years
    effective_strike = G / A  # Strike in terms of account performance
    
    # Since account follows stock (adjusted for fees), we can price as put on stock
    # with adjusted strike accounting for fee impact
    remaining_fees = np.exp(-gmab_params.fee_annual * T)
    adjusted_strike = effective_strike / remaining_fees
    
    if model == 'bs':
        # Black-Scholes put value = Call(K) - S + K*exp(-rT)  [put-call parity]
        call_value = bs_price_call(S, adjusted_strike, T, r, q, 0.2)  # Default vol = 20%
        put_value = call_value - S * np.exp(-q * T) + adjusted_strike * np.exp(-r * T)
        
        # Delta calculation via bump-and-revalue
        S_up = S * (1 + bump_size)
        call_value_up = bs_price_call(S_up, adjusted_strike, T, r, q, 0.2)
        put_value_up = call_value_up - S_up * np.exp(-q * T) + adjusted_strike * np.exp(-r * T)
        
        delta = (put_value_up - put_value) / (S_up - S)
        
    elif model == 'heston':
        if heston_params is None:
            raise ValueError("Heston parameters required for Heston pricing")
        
        # Heston call value
        call_value = carr_madan_call_price(
            S, adjusted_strike, T, r, q,
            heston_params['v0'], heston_params['kappa'], heston_params['theta'],
            heston_params['sigma_v'], heston_params['rho']
        )
        put_value = call_value - S * np.exp(-q * T) + adjusted_strike * np.exp(-r * T)
        
        # Delta via bump-and-revalue
        S_up = S * (1 + bump_size)
        call_value_up = carr_madan_call_price(
            S_up, adjusted_strike, T, r, q,
            heston_params['v0'], heston_params['kappa'], heston_params['theta'],
            heston_params['sigma_v'], heston_params['rho']
        )
        put_value_up = call_value_up - S_up * np.exp(-q * T) + adjusted_strike * np.exp(-r * T)
        
        delta = (put_value_up - put_value) / (S_up - S)
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Scale by lambda and account size
    scaled_value = lambda_scale * A * max(put_value, 0.0)
    scaled_delta = lambda_scale * A * delta
    
    return scaled_value, scaled_delta

def gmab_maturity_payoff(A_T: float, gmab_params: GMABParams, initial_premium: float) -> float:
    """
    Calculate GMAB payoff at maturity.
    
    Args:
        A_T: Account value at maturity (not normalized)
        gmab_params: GMAB product parameters
        initial_premium: Initial premium paid by policyholder
    
    Returns:
        payoff: Top-up amount = max(G - A_T, 0) where G is the guaranteed amount
    """
    G = initial_premium * (1 + gmab_params.g_annual) ** gmab_params.T_years
    payoff = max(G - A_T, 0.0)  # Fixed: G - A_T, not G - initial_premium * A_T
    return payoff

if __name__ == "__main__":
    # Test GMAB functionality
    params = GMABParams()
    
    # Test evolve_account_from_prices
    n_steps, n_paths = 252, 1000
    S = np.random.randn(n_steps + 1, n_paths).cumsum(axis=0)
    S = 4500 * np.exp(S * 0.2 / np.sqrt(252))  # GBM-like paths
    
    dt_years = 1.0 / 252
    A = evolve_account_from_prices(S, params.fee_annual, dt_years)
    
    print(f"Initial S: {S[0, 0]:.2f}, Final S: {S[-1, 0]:.2f}")
    print(f"Initial A: {A[0, 0]:.2f}, Final A: {A[-1, 0]:.2f}")
    
    # Test GMAB valuation
    value, delta = gmab_value_and_delta(
        S=4500, A=4500, T=1.0, r=0.02, q=0.01, 
        gmab_params=params, model='bs'
    )
    print(f"GMAB Value: {value:.2f}, Delta: {delta:.4f}")
    
    # Test maturity payoff
    payoff = gmab_maturity_payoff(A[-1, 0] / S[0, 0], params, 1.0)
    print(f"Maturity payoff (normalized): {payoff:.4f}")