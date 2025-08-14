"""
RILA (Registered Index-Linked Annuity) Payoff Implementation

This module implements RILA payoff functions and static replication methods for
efficient valuation and hedging under stochastic volatility models.

Author: Abdurakhmonbek Fayzullaev
Purpose: MSc Dissertation - Solvency II SCR for Equity-Linked Variable Annuities
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional
from scipy.optimize import minimize_scalar
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rila_payoff(ST: np.ndarray, S0: float, cap: float, buffer: float) -> np.ndarray:
    """
    Compute the terminal payoff for a RILA (Registered Index-Linked Annuity) product.
    
    A RILA provides downside protection through a buffer and limits upside through a cap.
    The payoff structure is piecewise linear:
    - If return < -buffer: participant loses more than buffer amount
    - If -buffer ≤ return ≤ 0: buffer absorbs losses, participant receives initial investment
    - If 0 < return ≤ cap: participant receives full upside
    - If return > cap: upside is capped at cap level
    
    Args:
        ST (np.ndarray): Terminal asset prices (vectorized)
        S0 (float): Initial asset price
        cap (float): Maximum participation rate (e.g., 0.25 for 25% cap)
        buffer (float): Downside protection level (e.g., 0.10 for 10% buffer)
        
    Returns:
        np.ndarray: RILA payoffs normalized to initial investment of 1.0
    """
    
    # Calculate total returns
    returns = (ST - S0) / S0
    
    # Initialize payoffs (all start at 1.0 = initial investment)
    payoffs = np.ones_like(returns)
    
    # Apply RILA payoff structure
    # Case 1: Losses beyond buffer (return < -buffer)
    severe_loss_mask = returns < -buffer
    payoffs[severe_loss_mask] = 1.0 + returns[severe_loss_mask] + buffer
    
    # Case 2: Losses within buffer (-buffer ≤ return ≤ 0) 
    # Buffer protects: payoff remains 1.0 (no change needed)
    
    # Case 3: Positive returns up to cap (0 < return ≤ cap)
    positive_uncapped_mask = (returns > 0) & (returns <= cap)
    payoffs[positive_uncapped_mask] = 1.0 + returns[positive_uncapped_mask]
    
    # Case 4: Returns above cap (return > cap)
    capped_mask = returns > cap
    payoffs[capped_mask] = 1.0 + cap
    
    # Ensure payoffs are non-negative (floor at zero)
    payoffs = np.maximum(payoffs, 0.0)
    
    return payoffs

def rila_replication(S0: float, K_grid: np.ndarray, cap: float, buffer: float, 
                    params: Dict) -> Dict[str, np.ndarray]:
    """
    Construct static vanilla option replication for RILA payoff.
    
    The RILA payoff is piecewise linear and can be replicated using vanilla puts and calls
    at specific strike levels. This enables fast re-valuation without nested Monte Carlo.
    
    RILA payoff structure:
    - Long position in underlying (1.0 shares)
    - Long put at strike K_buffer = S0 * (1 - buffer) with coefficient = 1.0
    - Short call at strike K_cap = S0 * (1 + cap) with coefficient = 1.0
    - Adjustments for the piecewise nature
    
    Args:
        S0 (float): Initial asset price
        K_grid (np.ndarray): Strike prices for available options
        cap (float): Maximum participation rate
        buffer (float): Downside protection level
        params (Dict): Model parameters for option pricing
        
    Returns:
        Dict containing:
            'weights': Option weights for replication
            'strikes': Corresponding strike prices
            'types': Option types ('call' or 'put')
            'underlying_weight': Position in underlying asset
    """
    
    # Calculate critical strike prices
    K_buffer = S0 * (1 - buffer)  # Put strike for buffer protection
    K_cap = S0 * (1 + cap)        # Call strike for cap limitation
    
    # Find closest available strikes
    buffer_strike_idx = np.argmin(np.abs(K_grid - K_buffer))
    cap_strike_idx = np.argmin(np.abs(K_grid - K_cap))
    
    K_buffer_actual = K_grid[buffer_strike_idx]
    K_cap_actual = K_grid[cap_strike_idx]
    
    # Log the replication setup
    logger.info(f"RILA Replication Setup:")
    logger.info(f"  S0: {S0:.2f}")
    logger.info(f"  Buffer: {buffer:.1%} -> Put strike: {K_buffer:.2f} (actual: {K_buffer_actual:.2f})")
    logger.info(f"  Cap: {cap:.1%} -> Call strike: {K_cap:.2f} (actual: {K_cap_actual:.2f})")
    
    # Construct replication portfolio
    replication = {
        'underlying_weight': 1.0,  # Long 1 share of underlying
        'strikes': np.array([K_buffer_actual, K_cap_actual]),
        'weights': np.array([1.0, -1.0]),  # Long put, short call
        'types': np.array(['put', 'call']),
        'K_buffer': K_buffer_actual,
        'K_cap': K_cap_actual,
        'buffer': buffer,
        'cap': cap
    }
    
    return replication

def validate_rila_replication(S0: float, cap: float, buffer: float, 
                             replication: Dict, ST_test: np.ndarray,
                             option_pricer: callable) -> Dict[str, float]:
    """
    Validate the RILA replication by comparing direct payoff calculation
    with replication portfolio payoff.
    
    Args:
        S0 (float): Initial asset price
        cap (float): Cap level
        buffer (float): Buffer level  
        replication (Dict): Replication portfolio from rila_replication()
        ST_test (np.ndarray): Test terminal prices
        option_pricer (callable): Function to price vanilla options
        
    Returns:
        Dict with validation metrics
    """
    
    # Calculate direct RILA payoffs
    direct_payoffs = rila_payoff(ST_test, S0, cap, buffer)
    
    # Calculate replication payoffs
    replication_payoffs = replication['underlying_weight'] * ST_test / S0
    
    # Add option payoffs
    for i, (strike, weight, option_type) in enumerate(
        zip(replication['strikes'], replication['weights'], replication['types'])
    ):
        if option_type == 'call':
            option_payoffs = np.maximum(ST_test - strike, 0) * weight / S0
        else:  # put
            option_payoffs = np.maximum(strike - ST_test, 0) * weight / S0
            
        replication_payoffs += option_payoffs
    
    # Calculate validation metrics
    abs_error = np.abs(direct_payoffs - replication_payoffs)
    rel_error = abs_error / np.maximum(direct_payoffs, 1e-8)
    
    validation_metrics = {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'rmse': np.sqrt(np.mean(abs_error**2))
    }
    
    logger.info("RILA Replication Validation:")
    for metric, value in validation_metrics.items():
        if 'rel_error' in metric:
            logger.info(f"  {metric}: {value:.2%}")
        else:
            logger.info(f"  {metric}: {value:.6f}")
    
    return validation_metrics

def rila_pv(S0: float, T: float, r: float, q: float, cap: float, buffer: float,
           option_pricer: callable, **pricer_params) -> float:
    """
    Calculate the present value of a RILA contract using static replication.
    
    Args:
        S0 (float): Current asset price
        T (float): Time to maturity
        r (float): Risk-free rate
        q (float): Dividend yield
        cap (float): Cap level
        buffer (float): Buffer level
        option_pricer (callable): Vanilla option pricing function
        **pricer_params: Additional parameters for option pricer
        
    Returns:
        float: Present value of RILA contract
    """
    
    # Create strike grid around critical levels
    K_buffer = S0 * (1 - buffer)
    K_cap = S0 * (1 + cap)
    K_grid = np.array([K_buffer, K_cap])
    
    # Get replication portfolio
    replication = rila_replication(S0, K_grid, cap, buffer, pricer_params)
    
    # Calculate PV components
    pv_underlying = S0 * np.exp(-q * T)  # Forward price of underlying
    pv_options = 0.0
    
    # Price the options in the replication portfolio
    for strike, weight, option_type in zip(
        replication['strikes'], replication['weights'], replication['types']
    ):
        option_pv = option_pricer(
            S0=S0, K=strike, T=T, r=r, q=q, 
            option_type=option_type, **pricer_params
        )
        pv_options += weight * option_pv
        
        logger.debug(f"Option: {option_type} K={strike:.2f}, weight={weight:.3f}, PV={option_pv:.6f}")
    
    # Total RILA PV
    rila_pv_total = replication['underlying_weight'] * pv_underlying + pv_options
    
    logger.info(f"RILA PV Components:")
    logger.info(f"  Underlying PV: {pv_underlying:.6f}")
    logger.info(f"  Options PV: {pv_options:.6f}")
    logger.info(f"  Total RILA PV: {rila_pv_total:.6f}")
    
    return rila_pv_total

def rila_greeks(S0: float, T: float, r: float, q: float, cap: float, buffer: float,
               option_pricer: callable, bump_size: float = 0.01, **pricer_params) -> Dict[str, float]:
    """
    Calculate Greeks for RILA contract using finite differences.
    
    Args:
        S0 (float): Current asset price
        T (float): Time to maturity  
        r (float): Risk-free rate
        q (float): Dividend yield
        cap (float): Cap level
        buffer (float): Buffer level
        option_pricer (callable): Vanilla option pricing function
        bump_size (float): Relative bump size for finite differences
        **pricer_params: Additional parameters for option pricer
        
    Returns:
        Dict containing delta, gamma, theta, vega
    """
    
    # Base PV
    pv_base = rila_pv(S0, T, r, q, cap, buffer, option_pricer, **pricer_params)
    
    # Delta (sensitivity to underlying price)
    pv_up = rila_pv(S0 * (1 + bump_size), T, r, q, cap, buffer, option_pricer, **pricer_params)
    pv_down = rila_pv(S0 * (1 - bump_size), T, r, q, cap, buffer, option_pricer, **pricer_params)
    delta = (pv_up - pv_down) / (2 * S0 * bump_size)
    
    # Gamma (second derivative w.r.t. underlying)
    gamma = (pv_up + pv_down - 2 * pv_base) / (S0 * bump_size)**2
    
    # Theta (time decay)
    if T > 1/252:  # At least 1 day to maturity
        pv_theta = rila_pv(S0, T - 1/252, r, q, cap, buffer, option_pricer, **pricer_params)
        theta = pv_theta - pv_base
    else:
        theta = 0.0
    
    # Vega (volatility sensitivity) - if volatility parameter exists
    vega = 0.0
    if 'vol' in pricer_params or 'sigma' in pricer_params:
        vol_param = pricer_params.get('vol') or pricer_params.get('sigma')
        if vol_param is not None:
            vol_bump = vol_param * 0.01  # 1% relative bump
            
            params_up = pricer_params.copy()
            params_down = pricer_params.copy()
            
            if 'vol' in pricer_params:
                params_up['vol'] = vol_param + vol_bump
                params_down['vol'] = vol_param - vol_bump
            else:
                params_up['sigma'] = vol_param + vol_bump
                params_down['sigma'] = vol_param - vol_bump
            
            pv_vol_up = rila_pv(S0, T, r, q, cap, buffer, option_pricer, **params_up)
            pv_vol_down = rila_pv(S0, T, r, q, cap, buffer, option_pricer, **params_down)
            vega = (pv_vol_up - pv_vol_down) / (2 * vol_bump)
    
    greeks = {
        'delta': delta,
        'gamma': gamma, 
        'theta': theta,
        'vega': vega,
        'pv': pv_base
    }
    
    logger.info("RILA Greeks:")
    for greek, value in greeks.items():
        logger.info(f"  {greek}: {value:.6f}")
    
    return greeks

# Example usage and testing
if __name__ == "__main__":
    # Example RILA parameters
    S0 = 4500.0
    cap = 0.25      # 25% cap
    buffer = 0.10   # 10% buffer
    
    # Test terminal prices
    ST_test = np.array([3500, 4000, 4500, 5000, 5500, 6000])
    
    # Calculate and display payoffs
    payoffs = rila_payoff(ST_test, S0, cap, buffer)
    
    print("RILA Payoff Examples:")
    print("ST\tReturn\tPayoff\tGain/Loss")
    for i, (st, payoff) in enumerate(zip(ST_test, payoffs)):
        ret = (st - S0) / S0
        gain_loss = payoff - 1.0
        print(f"{st:.0f}\t{ret:.1%}\t{payoff:.3f}\t{gain_loss:+.3f}")
    
    # Test replication
    K_grid = np.arange(3000, 7000, 100)  # Strike grid
    replication = rila_replication(S0, K_grid, cap, buffer, {})
    
    print(f"\nReplication Portfolio:")
    print(f"Underlying weight: {replication['underlying_weight']:.3f}")
    for strike, weight, opt_type in zip(replication['strikes'], replication['weights'], replication['types']):
        print(f"{opt_type.upper()} K={strike:.0f}, weight={weight:+.3f}")