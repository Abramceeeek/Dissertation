import numpy as np
from scipy.stats import norm
from rila.payoff import apply_rila_payoff
from typing import Any, Dict

def black_scholes_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str = 'call') -> float:
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

def rila_delta_approximation(S: float, S0: float, T: float, r: float, q: float, sigma: float, buffer: float, cap: float) -> float:
    K_buffer = S0 * (1 - buffer)
    K_cap = S0 * (1 + cap)
    if T < 1/252:
        # Support vectorized S
        return np.where(np.asarray(S) > K_buffer, 1.0, 0.0)
    d1_buffer = (np.log(np.asarray(S)/K_buffer) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d1_cap = (np.log(np.asarray(S)/K_cap) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return 1 + norm.cdf(-d1_buffer) - norm.cdf(d1_cap)

def conditional_tail_expectation(values: np.ndarray, alpha: float) -> float:
    var = np.percentile(values, 100*alpha)
    tail = values[values <= var]
    return np.mean(tail) if len(tail) > 0 else var

def simulate_dynamic_hedge(price_paths: np.ndarray, S0: float, r: float, q: float, sigma: float, buffer: float = 0.1, cap: float = 0.5, rebalance_freq: int = 1, transaction_cost: float = 0.0) -> Any:
    n_steps, n_paths = price_paths.shape
    n_steps -= 1
    T_total = 7.0
    dt = T_total / n_steps
    hedge_portfolio_value = np.zeros((n_steps + 1, n_paths))
    hedge_shares = np.zeros((n_steps + 1, n_paths))
    cash_account = np.zeros((n_steps + 1, n_paths))
    initial_liability_value = S0
    T_remaining = T_total
    initial_delta = rila_delta_approximation(S0, S0, T_remaining, r, q, sigma, buffer, cap)
    hedge_shares[0, :] = initial_delta
    cash_account[0, :] = initial_liability_value - initial_delta * S0
    hedge_portfolio_value[0, :] = hedge_shares[0, :] * S0 + cash_account[0, :]
    for t in range(1, n_steps + 1):
        T_remaining = T_total - t * dt
        S_current = price_paths[t, :]
        if t % rebalance_freq == 0 and T_remaining > 0:
            new_delta = rila_delta_approximation(S_current, S0, T_remaining, r, q, sigma, buffer, cap)
            shares_to_trade = new_delta - hedge_shares[t-1, :]
            trade_cost = np.abs(shares_to_trade) * S_current * transaction_cost
            hedge_shares[t, :] = new_delta
            cash_account[t, :] = (cash_account[t-1, :] * np.exp(r * dt) - shares_to_trade * S_current - trade_cost)
        else:
            hedge_shares[t, :] = hedge_shares[t-1, :]
            cash_account[t, :] = cash_account[t-1, :] * np.exp(r * dt)
        hedge_portfolio_value[t, :] = hedge_shares[t, :] * S_current + cash_account[t, :]
    final_returns = (price_paths[-1, :] - S0) / S0
    credited_returns = apply_rila_payoff(final_returns, buffer, cap)
    final_liability_payoff = S0 * (1 + credited_returns)
    final_hedge_value = hedge_portfolio_value[-1, :]
    hedge_pnl = final_hedge_value - final_liability_payoff
    return hedge_pnl, hedge_portfolio_value

def analyze_hedging_performance(hedge_pnl: np.ndarray, unhedged_pnl: np.ndarray = None) -> Dict[str, float]:
    hedge_pnl = np.array(hedge_pnl)
    stats = {
        'mean_pnl': np.mean(hedge_pnl),
        'std_pnl': np.std(hedge_pnl),
        'var_95': np.percentile(hedge_pnl, 5),
        'var_99': np.percentile(hedge_pnl, 1),
        'cte_95': conditional_tail_expectation(hedge_pnl, 0.05),
        'cte_99': conditional_tail_expectation(hedge_pnl, 0.01),
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