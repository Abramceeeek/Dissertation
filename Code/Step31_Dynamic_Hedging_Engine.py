"""
Dynamic Hedging Engine for RILA Products under Stochastic Volatility Models

This module implements dynamic hedging strategies for Registered Index-Linked Annuities (RILAs)
under Heston and Rough Volatility models for Solvency II capital requirement analysis.

Author: Abdurakhmonbek Fayzullaev
Purpose: MSc Dissertation - Solvency II SCR for Equity-Linked Variable Annuities
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Callable, Optional
from scipy.interpolate import interp1d
import logging
from Step23_RILA_Payoff_and_Replication import rila_pv, rila_greeks, rila_replication
from Step16_Heston_Carr_Madan_Pricing import heston_call_price
from Step17_Heston_Pricing_Utilities import heston_put_price

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rebalance_schedule(t_grid: np.ndarray, rebalance: str) -> np.ndarray:
    """
    Generate rebalancing time indices based on frequency.
    
    Args:
        t_grid (np.ndarray): Time grid in years
        rebalance (str): "daily", "weekly", or "monthly"
        
    Returns:
        np.ndarray: Boolean mask for rebalancing times
    """
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 1/252
    
    if rebalance == "daily":
        # Every day
        return np.ones(len(t_grid), dtype=bool)
    elif rebalance == "weekly":
        # Every 5 business days (weekly)
        freq = max(1, int(5 * dt * 252))  # 5 days
        mask = np.zeros(len(t_grid), dtype=bool)
        mask[::freq] = True
        mask[0] = True  # Always include initial time
        return mask
    elif rebalance == "monthly":
        # Every 21 business days (monthly)
        freq = max(1, int(21 * dt * 252))  # 21 days
        mask = np.zeros(len(t_grid), dtype=bool)
        mask[::freq] = True
        mask[0] = True  # Always include initial time
        return mask
    else:
        raise ValueError(f"Unknown rebalancing frequency: {rebalance}")

def create_heston_option_pricer(heston_params: Dict) -> Callable:
    """
    Create option pricing function for Heston model.
    
    Args:
        heston_params (Dict): Heston model parameters
        
    Returns:
        Callable: Option pricing function
    """
    def price_option(S0: float, K: float, T: float, r: float, q: float, 
                    option_type: str = 'call', **kwargs) -> float:
        """Price vanilla option under Heston model."""
        v0 = heston_params['v0']
        kappa = heston_params['kappa'] 
        theta = heston_params['theta']
        sigma_v = heston_params['sigma_v']
        rho = heston_params['rho']
        
        if option_type.lower() == 'call':
            return heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
        elif option_type.lower() == 'put':
            return heston_put_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
    
    return price_option

def create_roughvol_option_pricer(roughvol_params: Dict) -> Callable:
    """
    Create option pricing function for Rough Volatility model.
    
    Note: This uses Heston approximation as specified in requirements.
    
    Args:
        roughvol_params (Dict): Rough volatility parameters
        
    Returns:
        Callable: Option pricing function (using Heston proxy)
    """
    # Map rough vol parameters to Heston equivalent
    # This is a simplified approximation as noted in requirements
    heston_proxy = {
        'v0': roughvol_params.get('v0', 0.04),
        'kappa': roughvol_params.get('lambda', 2.0),  # Mean reversion
        'theta': roughvol_params.get('v0', 0.04),     # Long-term var
        'sigma_v': roughvol_params.get('nu', 0.3),    # Vol of vol
        'rho': roughvol_params.get('rho', -0.7)       # Correlation
    }
    
    logger.warning("Using Heston proxy approximation for Rough Volatility option pricing")
    return create_heston_option_pricer(heston_proxy)

def interpolate_curve(curve_data: Union[Callable, np.ndarray, float], 
                     times: np.ndarray) -> np.ndarray:
    """
    Interpolate rate/dividend curve to specific times.
    
    Args:
        curve_data: Callable, array, or constant rate
        times: Time points for interpolation
        
    Returns:
        np.ndarray: Interpolated rates
    """
    if callable(curve_data):
        return curve_data(times)
    elif isinstance(curve_data, (int, float)):
        return np.full_like(times, curve_data)
    elif isinstance(curve_data, np.ndarray):
        if len(curve_data) == len(times):
            return curve_data
        else:
            # Simple linear interpolation
            t_curve = np.linspace(0, times[-1], len(curve_data))
            interpolator = interp1d(t_curve, curve_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            return interpolator(times)
    else:
        raise ValueError(f"Unsupported curve data type: {type(curve_data)}")

def run_dynamic_hedge(
    model: str,
    S_paths: np.ndarray,
    t_grid: np.ndarray,
    r_curve: Union[Callable, np.ndarray, float],
    q_curve: Union[Callable, np.ndarray, float],
    rila_params: Dict,
    rebalance: str = "weekly",
    trans_cost_bps: float = 1.0,
    hedge_instruments: Dict = {"delta": True, "vega": False},
    seed: int = 42,
) -> Dict:
    """
    Run dynamic hedging simulation for RILA products.
    
    Args:
        model (str): "heston" or "roughvol"
        S_paths (np.ndarray): Simulated price paths (n_paths, n_steps+1)
        t_grid (np.ndarray): Time grid in years
        r_curve: Risk-free rate curve (callable, array, or constant)
        q_curve: Dividend yield curve (callable, array, or constant)
        rila_params (Dict): RILA product parameters
        rebalance (str): "daily", "weekly", or "monthly"
        trans_cost_bps (float): Transaction cost in basis points
        hedge_instruments (Dict): Hedging instrument flags
        seed (int): Random seed
        
    Returns:
        Dict containing hedge results
    """
    
    np.random.seed(seed)
    n_paths, n_steps = S_paths.shape[0], S_paths.shape[1] - 1
    
    logger.info(f"Starting dynamic hedging simulation:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Paths: {n_paths:,}")
    logger.info(f"  Time steps: {n_steps}")
    logger.info(f"  Rebalance: {rebalance}")
    logger.info(f"  Transaction cost: {trans_cost_bps:.1f} bps")
    
    # Extract RILA parameters
    S0 = rila_params['S0']
    T = rila_params['T']
    cap = rila_params['cap']
    buffer = rila_params['buffer']
    
    # Get model parameters
    if model.lower() == 'heston':
        model_params = rila_params.get('heston_params', {})
        option_pricer = create_heston_option_pricer(model_params)
    elif model.lower() == 'roughvol':
        model_params = rila_params.get('roughvol_params', {})
        option_pricer = create_roughvol_option_pricer(model_params)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Interpolate rate curves
    r_rates = interpolate_curve(r_curve, t_grid)
    q_rates = interpolate_curve(q_curve, t_grid)
    
    # Get rebalancing schedule
    rebalance_mask = get_rebalance_schedule(t_grid, rebalance)
    rebalance_times = np.where(rebalance_mask)[0]
    
    logger.info(f"Rebalancing at {len(rebalance_times)} time points")
    
    # Initialize tracking arrays
    hedge_pnl = np.zeros(n_paths)
    hedge_error = np.zeros(n_paths)
    transaction_costs = np.zeros(n_paths)
    
    # Track portfolio positions
    delta_positions = np.zeros(n_paths)  # Delta hedge position
    vega_positions = np.zeros(n_paths)   # Vega hedge position (if enabled)
    cash_positions = np.zeros(n_paths)   # Cash account
    
    # Path-wise diagnostics
    pathwise_data = {
        'liability_values': np.zeros((n_paths, len(rebalance_times))),
        'hedge_positions': np.zeros((n_paths, len(rebalance_times))),
        'portfolio_values': np.zeros((n_paths, len(rebalance_times))),
        'transaction_costs': np.zeros((n_paths, len(rebalance_times)))
    }
    
    # Main hedging loop
    for i, t_idx in enumerate(rebalance_times):
        t = t_grid[t_idx]
        time_to_maturity = T - t
        
        if time_to_maturity <= 0:
            break
            
        S_current = S_paths[:, t_idx]
        r_current = r_rates[t_idx]
        q_current = q_rates[t_idx]
        
        # Calculate current liability value and Greeks for each path
        for path in range(n_paths):
            S_path = S_current[path]
            
            try:
                # Calculate RILA Greeks
                greeks = rila_greeks(
                    S0=S_path, T=time_to_maturity, r=r_current, q=q_current,
                    cap=cap, buffer=buffer, option_pricer=option_pricer,
                    **model_params
                )
                
                current_delta = greeks['delta']
                current_vega = greeks.get('vega', 0.0) if hedge_instruments.get('vega', False) else 0.0
                current_liability_pv = greeks['pv']
                
                # Calculate required position changes
                delta_change = current_delta - delta_positions[path]
                vega_change = current_vega - vega_positions[path] if hedge_instruments.get('vega', False) else 0.0
                
                # Calculate transaction costs
                if i > 0:  # No transaction costs at initial setup
                    tc_delta = abs(delta_change) * S_path * (trans_cost_bps / 10000)
                    tc_vega = abs(vega_change) * S_path * (trans_cost_bps / 10000) * 0.1  # Scaled for vega hedge
                    path_tc = tc_delta + tc_vega
                    transaction_costs[path] += path_tc
                    pathwise_data['transaction_costs'][path, i] = path_tc
                
                # Update positions
                delta_positions[path] = current_delta
                if hedge_instruments.get('vega', False):
                    vega_positions[path] = current_vega
                
                # Track values
                pathwise_data['liability_values'][path, i] = current_liability_pv
                pathwise_data['hedge_positions'][path, i] = delta_positions[path] * S_path
                pathwise_data['portfolio_values'][path, i] = (
                    pathwise_data['hedge_positions'][path, i] - current_liability_pv
                )
                
            except Exception as e:
                logger.warning(f"Error in path {path} at time {t:.3f}: {e}")
                continue
    
    # Calculate final hedge P&L and errors
    final_time_idx = min(len(t_grid) - 1, rebalance_times[-1])
    
    for path in range(n_paths):
        try:
            # Final asset price
            S_final = S_paths[path, final_time_idx]
            
            # Final hedge portfolio value
            final_hedge_value = delta_positions[path] * S_final
            if hedge_instruments.get('vega', False):
                # Approximate vega hedge contribution (simplified)
                final_hedge_value += vega_positions[path] * 0.1 * S_final
            
            # Initial liability value
            initial_liability = rila_pv(
                S0=S0, T=T, r=r_rates[0], q=q_rates[0],
                cap=cap, buffer=buffer, option_pricer=option_pricer,
                **model_params
            )
            
            # Final liability value  
            time_to_final = max(0, T - t_grid[final_time_idx])
            if time_to_final > 0:
                final_liability = rila_pv(
                    S0=S_final, T=time_to_final, r=r_rates[final_time_idx], q=q_rates[final_time_idx],
                    cap=cap, buffer=buffer, option_pricer=option_pricer,
                    **model_params
                )
            else:
                # At maturity
                from rila_payoff import rila_payoff
                final_liability = rila_payoff(np.array([S_final]), S0, cap, buffer)[0]
            
            # Hedge P&L = change in hedge portfolio value - transaction costs
            initial_hedge_value = delta_positions[0] * S0 if len(delta_positions) > 0 else 0.0
            hedge_pnl[path] = final_hedge_value - initial_hedge_value - transaction_costs[path]
            
            # Hedge error = change in liability - hedge P&L
            liability_change = final_liability - initial_liability
            hedge_error[path] = liability_change - hedge_pnl[path]
            
        except Exception as e:
            logger.warning(f"Error calculating final P&L for path {path}: {e}")
            hedge_pnl[path] = 0.0
            hedge_error[path] = 0.0
    
    # Calculate trade statistics
    trade_stats = {
        'total_rebalances': len(rebalance_times),
        'avg_transaction_cost': np.mean(transaction_costs),
        'total_transaction_cost': np.sum(transaction_costs),
        'max_transaction_cost': np.max(transaction_costs),
        'avg_hedge_error': np.mean(np.abs(hedge_error)),
        'hedge_error_std': np.std(hedge_error)
    }
    
    # Results summary
    results = {
        'hedge_pnl': hedge_pnl,
        'hedge_error': hedge_error,
        'transaction_costs': transaction_costs,
        'trade_stats': trade_stats,
        'pathwise': pathwise_data,
        'config': {
            'model': model,
            'rebalance': rebalance,
            'trans_cost_bps': trans_cost_bps,
            'hedge_instruments': hedge_instruments,
            'n_paths': n_paths,
            'n_rebalances': len(rebalance_times)
        }
    }
    
    # Log summary statistics
    logger.info("Dynamic Hedging Results:")
    logger.info(f"  Mean hedge P&L: {np.mean(hedge_pnl):,.2f}")
    logger.info(f"  Hedge P&L std: {np.std(hedge_pnl):,.2f}")
    logger.info(f"  Mean hedge error: {np.mean(np.abs(hedge_error)):,.2f}")
    logger.info(f"  Avg transaction cost: {np.mean(transaction_costs):,.2f}")
    logger.info(f"  Total rebalances: {trade_stats['total_rebalances']}")
    
    return results

def analyze_hedge_effectiveness(hedge_results: Dict, unhedged_pnl: np.ndarray) -> Dict:
    """
    Analyze the effectiveness of the dynamic hedging strategy.
    
    Args:
        hedge_results (Dict): Results from run_dynamic_hedge()
        unhedged_pnl (np.ndarray): Unhedged P&L for comparison
        
    Returns:
        Dict: Hedge effectiveness metrics
    """
    
    hedged_pnl = hedge_results['hedge_pnl'] + unhedged_pnl
    hedge_error = hedge_results['hedge_error']
    
    # Risk reduction metrics
    unhedged_var = np.var(unhedged_pnl)
    hedged_var = np.var(hedged_pnl)
    hedge_error_var = np.var(hedge_error)
    
    variance_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var > 0 else 0.0
    hedge_efficiency = 1 - (hedge_error_var / unhedged_var) if unhedged_var > 0 else 0.0
    
    # Risk metrics
    effectiveness_metrics = {
        'variance_reduction': variance_reduction,
        'hedge_efficiency': hedge_efficiency,
        'unhedged_std': np.sqrt(unhedged_var),
        'hedged_std': np.sqrt(hedged_var),
        'hedge_error_std': np.sqrt(hedge_error_var),
        'correlation': np.corrcoef(unhedged_pnl, hedge_results['hedge_pnl'])[0,1] if len(unhedged_pnl) > 1 else 0.0,
        'avg_transaction_cost_pct': np.mean(hedge_results['transaction_costs']) / np.mean(np.abs(unhedged_pnl)) * 100
    }
    
    logger.info("Hedge Effectiveness Analysis:")
    logger.info(f"  Variance reduction: {variance_reduction:.1%}")
    logger.info(f"  Hedge efficiency: {hedge_efficiency:.1%}")
    logger.info(f"  Unhedged volatility: {effectiveness_metrics['unhedged_std']:,.2f}")
    logger.info(f"  Hedged volatility: {effectiveness_metrics['hedged_std']:,.2f}")
    logger.info(f"  Transaction cost %: {effectiveness_metrics['avg_transaction_cost_pct']:.2f}%")
    
    return effectiveness_metrics

# Example usage
if __name__ == "__main__":
    # Example parameters
    rila_params = {
        'S0': 4500.0,
        'T': 1.0,  # 1 year for SCR calculation
        'cap': 0.25,
        'buffer': 0.10,
        'heston_params': {
            'v0': 0.04,
            'kappa': 2.0,
            'theta': 0.04,
            'sigma_v': 0.3,
            'rho': -0.7
        }
    }
    
    # Generate example paths (would normally come from simulation)
    np.random.seed(42)
    n_paths, n_steps = 1000, 252
    t_grid = np.linspace(0, 1, n_steps + 1)
    
    # Simple GBM paths for testing
    dt = 1 / n_steps
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = rila_params['S0']
    
    for i in range(n_steps):
        S_paths[:, i+1] = S_paths[:, i] * np.exp(
            (0.05 - 0.5 * 0.2**2) * dt + 0.2 * dW[:, i]
        )
    
    # Run hedging simulation
    results = run_dynamic_hedge(
        model='heston',
        S_paths=S_paths,
        t_grid=t_grid,
        r_curve=0.02,
        q_curve=0.01,
        rila_params=rila_params,
        rebalance='weekly',
        trans_cost_bps=1.0
    )
    
    print(f"\nExample Results:")
    print(f"Mean hedge P&L: ${np.mean(results['hedge_pnl']):.2f}")
    print(f"Hedge error std: ${np.std(results['hedge_error']):.2f}")
    print(f"Avg transaction cost: ${np.mean(results['transaction_costs']):.2f}")