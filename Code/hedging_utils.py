import numpy as np
import pandas as pd
from scipy.stats import norm
from utils import apply_rila_payoff, get_r_for_discounting

def black_scholes_delta(S, K, T, r, q, sigma, option_type='call'):
    """
    Calculate Black-Scholes delta for vanilla options.
    
    Parameters:
    - S: current stock price
    - K: strike price
    - T: time to maturity
    - r: risk-free rate
    - q: dividend yield
    - sigma: volatility
    - option_type: 'call' or 'put'
    
    Returns:
    - delta: option delta
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    if option_type == 'call':
        return np.exp(-q*T) * norm.cdf(d1)
    else:
        return np.exp(-q*T) * (norm.cdf(d1) - 1)

def rila_delta_approximation(S, S0, T, r, q, sigma, buffer=0.1, cap=0.5):
    """
    Approximate the delta of a RILA payoff by decomposing it into vanilla options.
    
    RILA payoff = min(max(S_T, (1-buffer)*S0), (1+cap)*S0)
    
    This can be approximated as:
    - Long position in underlying
    - Long put struck at (1-buffer)*S0 
    - Short call struck at (1+cap)*S0
    
    Parameters:
    - S: current stock price
    - S0: initial stock price
    - T: time to maturity
    - r: risk-free rate
    - q: dividend yield
    - sigma: volatility
    - buffer: downside buffer (e.g., 0.1 for 10%)
    - cap: upside cap (e.g., 0.5 for 50%)
    
    Returns:
    - total_delta: combined delta of the RILA position
    """
    if T <= 0:
        return 1.0  # At maturity, delta is 1 in the middle region
    
    # Strike prices
    K_put = (1 - buffer) * S0  # Buffer strike
    K_call = (1 + cap) * S0    # Cap strike
    
    # Individual deltas
    delta_underlying = 1.0
    delta_put = black_scholes_delta(S, K_put, T, r, q, sigma, 'put')
    delta_call = black_scholes_delta(S, K_call, T, r, q, sigma, 'call')
    
    # Combined delta: long underlying + long put - short call
    total_delta = delta_underlying + delta_put - delta_call
    
    return total_delta

def simulate_dynamic_hedge(price_paths, S0, r, q, sigma, buffer=0.1, cap=0.5, 
                          rebalance_freq=1, transaction_cost=0.0):
    """
    Simulate dynamic hedging of RILA guarantees.
    
    Parameters:
    - price_paths: 2D array of shape (n_steps+1, n_paths) with stock price paths
    - S0: initial stock price
    - r: risk-free rate
    - q: dividend yield
    - sigma: volatility for hedging (may differ from realized)
    - buffer: RILA buffer level
    - cap: RILA cap level
    - rebalance_freq: rebalancing frequency (1=daily, 5=weekly, 21=monthly)
    - transaction_cost: proportional transaction cost (e.g., 0.001 for 0.1%)
    
    Returns:
    - hedge_pnl: final hedging P&L for each path
    - hedge_errors: hedging errors over time
    """
    n_steps, n_paths = price_paths.shape
    n_steps -= 1  # Adjust for 0-indexing
    T_total = 7.0  # Total time in years
    dt = T_total / n_steps
    
    # Initialize arrays
    hedge_portfolio_value = np.zeros((n_steps + 1, n_paths))
    hedge_shares = np.zeros((n_steps + 1, n_paths))
    cash_account = np.zeros((n_steps + 1, n_paths))
    
    # Calculate liability fair value at inception (approximate)
    initial_liability_value = S0  # Simplified: assume fair value equals initial investment
    
    # Initialize hedge portfolio
    T_remaining = T_total
    initial_delta = rila_delta_approximation(S0, S0, T_remaining, r, q, sigma, buffer, cap)
    hedge_shares[0, :] = initial_delta
    cash_account[0, :] = initial_liability_value - initial_delta * S0
    hedge_portfolio_value[0, :] = hedge_shares[0, :] * S0 + cash_account[0, :]
    
    # Dynamic hedging simulation
    for t in range(1, n_steps + 1):
        T_remaining = T_total - t * dt
        
        # Current stock prices
        S_current = price_paths[t, :]
        
        # Calculate new hedge ratio
        if t % rebalance_freq == 0 and T_remaining > 0:
            new_delta = rila_delta_approximation(S_current, S0, T_remaining, r, q, sigma, buffer, cap)
            
            # Calculate rebalancing trades
            shares_to_trade = new_delta - hedge_shares[t-1, :]
            
            # Apply transaction costs
            trade_cost = np.abs(shares_to_trade) * S_current * transaction_cost
            
            # Update positions
            hedge_shares[t, :] = new_delta
            cash_account[t, :] = (cash_account[t-1, :] * np.exp(r * dt) - 
                                shares_to_trade * S_current - trade_cost)
        else:
            # No rebalancing - carry forward positions
            hedge_shares[t, :] = hedge_shares[t-1, :]
            cash_account[t, :] = cash_account[t-1, :] * np.exp(r * dt)
        
        # Mark-to-market hedge portfolio
        hedge_portfolio_value[t, :] = hedge_shares[t, :] * S_current + cash_account[t, :]
    
    # Calculate final RILA payoffs
    final_returns = (price_paths[-1, :] - S0) / S0
    credited_returns = apply_rila_payoff(final_returns, buffer, cap)
    final_liability_payoff = S0 * (1 + credited_returns)
    
    # Calculate hedging P&L
    final_hedge_value = hedge_portfolio_value[-1, :]
    hedge_pnl = final_hedge_value - final_liability_payoff
    
    return hedge_pnl, hedge_portfolio_value

def analyze_hedging_performance(hedge_pnl, unhedged_pnl=None):
    """
    Analyze the performance of dynamic hedging strategy.
    
    Parameters:
    - hedge_pnl: array of hedging P&L outcomes
    - unhedged_pnl: array of unhedged liability outcomes (optional)
    
    Returns:
    - performance_stats: dictionary with key risk metrics
    """
    hedge_pnl = np.array(hedge_pnl)
    
    stats = {
        'mean_pnl': np.mean(hedge_pnl),
        'std_pnl': np.std(hedge_pnl),
        'var_95': np.percentile(hedge_pnl, 5),
        'var_99': np.percentile(hedge_pnl, 1),
        'cte_95': np.mean(hedge_pnl[hedge_pnl <= np.percentile(hedge_pnl, 5)]),
        'cte_99': np.mean(hedge_pnl[hedge_pnl <= np.percentile(hedge_pnl, 1)]),
        'worst_case': np.min(hedge_pnl),
        'best_case': np.max(hedge_pnl),
        'prob_loss': np.mean(hedge_pnl < 0)
    }
    
    if unhedged_pnl is not None:
        unhedged_pnl = np.array(unhedged_pnl)
        stats['unhedged_var_95'] = np.percentile(unhedged_pnl, 5)
        stats['unhedged_var_99'] = np.percentile(unhedged_pnl, 1)
        stats['unhedged_std'] = np.std(unhedged_pnl)
        stats['risk_reduction_var95'] = (np.percentile(unhedged_pnl, 5) - stats['var_95']) / abs(np.percentile(unhedged_pnl, 5))
        stats['risk_reduction_std'] = (np.std(unhedged_pnl) - stats['std_pnl']) / np.std(unhedged_pnl)
    
    return stats