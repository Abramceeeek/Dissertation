"""
Step 14: Variable Annuity Simulation and Dynamic Hedging Analysis

This is the main script for the dissertation: "The Effects of Stochastic Volatility Models and Dynamic 
Hedging Strategies on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk 
Management Approach"

This script:
1. Simulates equity returns under different stochastic volatility models (GBM, Heston, Rough Vol)
2. Applies variable annuity payoff structures with guarantees
3. Implements dynamic hedging strategies
4. Calculates capital requirements under Solvency II framework
5. Compares risk metrics across different models and hedging strategies

Author: Abdurakhmonbek Fayzullaev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from VA_Configuration import *

def simulate_gbm_paths(S0, T, N, n_paths, r, sigma, q=0, seed=None):
    """
    Simulate stock price paths using Geometric Brownian Motion (GBM).
    
    This represents the simplest model for equity returns, assuming constant volatility.
    While unrealistic, it serves as a baseline for comparison with stochastic volatility models.
    
    Parameters:
        S0 (float): Initial stock price
        T (float): Time to maturity in years
        N (int): Number of time steps
        n_paths (int): Number of simulation paths
        r (float): Risk-free interest rate
        sigma (float): Constant volatility
        q (float): Dividend yield
        seed (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Array of simulated price paths (n_paths x N+1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Generate random shocks
    Z = np.random.normal(0, 1, (n_paths, N))
    
    # Initialize price paths
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    
    # Simulate price evolution
    for t in range(N):
        S[:, t + 1] = S[:, t] * np.exp(drift + diffusion * Z[:, t])
    
    return S

def simulate_heston_paths(S0, T, N, n_paths, r, v0, kappa, theta, sigma_v, rho, q=0, seed=None):
    """
    Simulate stock price paths using the Heston stochastic volatility model.
    
    The Heston model captures the volatility smile and allows volatility to be stochastic,
    which is more realistic than constant volatility assumptions. It includes:
    - Mean-reverting volatility
    - Correlation between stock returns and volatility (leverage effect)
    - Volatility clustering observed in financial markets
    
    Parameters:
        S0 (float): Initial stock price
        T (float): Time to maturity in years  
        N (int): Number of time steps
        n_paths (int): Number of simulation paths
        r (float): Risk-free interest rate
        v0 (float): Initial variance
        kappa (float): Mean reversion speed
        theta (float): Long-term variance level
        sigma_v (float): Volatility of volatility
        rho (float): Correlation between price and volatility
        q (float): Dividend yield
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (price_paths, variance_paths) - Arrays of simulated paths
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    
    # Generate correlated random variables
    Z1 = np.random.normal(0, 1, (n_paths, N))
    Z2 = np.random.normal(0, 1, (n_paths, N))
    
    # Apply correlation structure
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Initialize arrays
    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Euler discretization scheme
    for t in range(N):
        # Ensure variance stays positive (Feller condition handling)
        v_pos = np.maximum(v[:, t], 0.0001)
        
        # Update variance using Euler scheme
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos * dt) * W2[:, t]
        v[:, t + 1] = np.maximum(v[:, t + 1], 0.0001)  # Floor variance
        
        # Update stock price
        S[:, t + 1] = S[:, t] * np.exp(
            (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * W1[:, t]
        )
    
    return S, v

def calculate_variable_annuity_payoff(final_price, initial_account, guarantee_rate, T):
    """
    Calculate the payoff for a variable annuity with guaranteed minimum benefit.
    
    Variable annuities provide investors with:
    - Upside participation in equity market performance
    - Downside protection through guaranteed minimum benefits
    - This creates embedded put option value for the insurer
    
    Parameters:
        final_price (array): Final equity index values
        initial_account (float): Initial account value (premium paid)
        guarantee_rate (float): Guaranteed minimum annual return
        T (float): Contract maturity in years
        
    Returns:
        tuple: (account_values, guarantee_benefits, payoffs)
    """
    # Account value = initial investment × (final_index / initial_index)
    index_return = final_price / S0
    account_values = initial_account * index_return
    
    # Guaranteed benefit = initial account value compounded at guarantee rate
    guarantee_benefits = initial_account * (1 + guarantee_rate)**T
    
    # Final payoff = max(account value, guaranteed benefit)
    payoffs = np.maximum(account_values, guarantee_benefits)
    
    return account_values, guarantee_benefits, payoffs

def calculate_delta_hedge_pnl(price_paths, hedge_freq_days=1, transaction_cost=0.001):
    """
    Calculate profit/loss from delta hedging strategy for variable annuity guarantees.
    
    Delta hedging aims to neutralize the price risk of the embedded options in variable
    annuities by dynamically trading the underlying asset. This is critical for 
    insurance companies to manage their market risk exposure.
    
    Parameters:
        price_paths (array): Simulated price paths
        hedge_freq_days (int): Hedging frequency in days (1=daily, 7=weekly, etc.)
        transaction_cost (float): Proportional transaction cost per trade
        
    Returns:
        array: Hedging P&L for each path
    """
    n_paths, n_steps = price_paths.shape
    hedge_pnl = np.zeros(n_paths)
    
    hedge_intervals = hedge_freq_days
    hedge_points = np.arange(0, n_steps, hedge_intervals)
    
    for path_idx in range(n_paths):
        path_pnl = 0
        previous_delta = 0
        
        for i, step in enumerate(hedge_points[:-1]):
            current_price = price_paths[path_idx, step]
            next_step = hedge_points[i + 1] if i + 1 < len(hedge_points) else n_steps - 1
            next_price = price_paths[path_idx, next_step]
            
            # Calculate Black-Scholes delta for the guarantee (simplified)
            time_to_expiry = (n_steps - step) / 252  # Convert to years
            if time_to_expiry > 0.001:  # Avoid division by zero
                # Approximate delta for at-the-money guarantee
                delta = 0.5  # Simplified delta calculation
            else:
                delta = 0
            
            # Calculate hedge adjustment needed
            delta_change = delta - previous_delta
            
            # Trading cost
            if abs(delta_change) > 0.001:  # Only trade if significant change
                trade_cost = abs(delta_change) * current_price * transaction_cost
                path_pnl -= trade_cost
            
            # P&L from holding delta position
            position_pnl = previous_delta * (next_price - current_price)
            path_pnl += position_pnl
            
            previous_delta = delta
        
        hedge_pnl[path_idx] = path_pnl
    
    return hedge_pnl

def calculate_risk_metrics(payoffs, hedged_pnl=None, confidence_levels=[0.95, 0.99, 0.995]):
    """
    Calculate comprehensive risk metrics for capital requirement analysis.
    
    These metrics are used in Solvency II capital calculations and enterprise
    risk management frameworks for insurance companies.
    
    Parameters:
        payoffs (array): Unhedged payoff distribution
        hedged_pnl (array): Hedged P&L distribution (optional)
        confidence_levels (list): Confidence levels for VaR calculation
        
    Returns:
        dict: Dictionary containing various risk metrics
    """
    metrics = {}
    
    # Unhedged metrics
    metrics['mean_unhedged'] = np.mean(payoffs)
    metrics['std_unhedged'] = np.std(payoffs)
    metrics['min_unhedged'] = np.min(payoffs)
    metrics['max_unhedged'] = np.max(payoffs)
    
    # Calculate VaR and ES at different confidence levels
    for conf in confidence_levels:
        alpha = 1 - conf
        var_unhedged = np.percentile(payoffs, alpha * 100)
        es_unhedged = np.mean(payoffs[payoffs <= var_unhedged])
        
        metrics[f'var_{int(conf*100)}_unhedged'] = var_unhedged
        metrics[f'es_{int(conf*100)}_unhedged'] = es_unhedged
    
    # Hedged metrics (if hedging P&L provided)
    if hedged_pnl is not None:
        hedged_payoffs = payoffs + hedged_pnl
        
        metrics['mean_hedged'] = np.mean(hedged_payoffs)
        metrics['std_hedged'] = np.std(hedged_payoffs)
        metrics['min_hedged'] = np.min(hedged_payoffs)
        metrics['max_hedged'] = np.max(hedged_payoffs)
        
        # Hedging effectiveness
        metrics['volatility_reduction'] = (metrics['std_unhedged'] - metrics['std_hedged']) / metrics['std_unhedged']
        metrics['mean_improvement'] = metrics['mean_hedged'] - metrics['mean_unhedged']
        
        for conf in confidence_levels:
            alpha = 1 - conf
            var_hedged = np.percentile(hedged_payoffs, alpha * 100)
            es_hedged = np.mean(hedged_payoffs[hedged_payoffs <= var_hedged])
            
            metrics[f'var_{int(conf*100)}_hedged'] = var_hedged
            metrics[f'es_{int(conf*100)}_hedged'] = es_hedged
    
    return metrics

def run_comprehensive_analysis():
    """
    Run comprehensive analysis comparing different models and hedging strategies.
    
    This is the main function that orchestrates the entire analysis for the dissertation.
    """
    print("="*80)
    print("COMPREHENSIVE VARIABLE ANNUITY ANALYSIS")
    print("The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies")
    print("on Capital Requirements for Equity-Linked Variable Annuities")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Analysis parameters
    r = 0.02  # Risk-free rate (2%)
    q = 0.01  # Dividend yield (1%)
    sigma_gbm = 0.20  # GBM volatility (20%)
    
    results = {}
    
    # Model 1: Geometric Brownian Motion (Baseline)
    print("1. GEOMETRIC BROWNIAN MOTION SIMULATION")
    print("-" * 50)
    gbm_paths = simulate_gbm_paths(S0, T, N, n_paths, r, sigma_gbm, q, random_seed)
    
    # Calculate VA payoffs
    final_prices_gbm = gbm_paths[:, -1]
    account_vals_gbm, guarantee_vals_gbm, payoffs_gbm = calculate_variable_annuity_payoff(
        final_prices_gbm, initial_account, guarantee_rate, T
    )
    
    # Calculate hedging P&L
    hedge_pnl_gbm = calculate_delta_hedge_pnl(gbm_paths, hedge_freq_days=1, 
                                              transaction_cost=transaction_cost)
    
    # Risk metrics
    risk_metrics_gbm = calculate_risk_metrics(payoffs_gbm, hedge_pnl_gbm)
    results['GBM'] = risk_metrics_gbm
    
    print(f"Mean account value: ${np.mean(account_vals_gbm):,.2f}")
    print(f"Guarantee triggered: {np.sum(final_prices_gbm < S0):,} paths ({np.sum(final_prices_gbm < S0)/n_paths:.1%})")
    print(f"Hedging effectiveness: {risk_metrics_gbm['volatility_reduction']:.1%} volatility reduction")
    print()
    
    # Model 2: Heston Stochastic Volatility
    print("2. HESTON STOCHASTIC VOLATILITY SIMULATION")  
    print("-" * 50)
    heston_prices, heston_vols = simulate_heston_paths(
        S0, T, N, n_paths, r, **heston_params, q=q, seed=random_seed
    )
    
    final_prices_heston = heston_prices[:, -1]
    account_vals_heston, guarantee_vals_heston, payoffs_heston = calculate_variable_annuity_payoff(
        final_prices_heston, initial_account, guarantee_rate, T
    )
    
    hedge_pnl_heston = calculate_delta_hedge_pnl(heston_prices, hedge_freq_days=1,
                                                 transaction_cost=transaction_cost)
    
    risk_metrics_heston = calculate_risk_metrics(payoffs_heston, hedge_pnl_heston)
    results['Heston'] = risk_metrics_heston
    
    print(f"Mean account value: ${np.mean(account_vals_heston):,.2f}")
    print(f"Guarantee triggered: {np.sum(final_prices_heston < S0):,} paths ({np.sum(final_prices_heston < S0)/n_paths:.1%})")
    print(f"Hedging effectiveness: {risk_metrics_heston['volatility_reduction']:.1%} volatility reduction")
    print(f"Mean volatility: {np.mean(np.sqrt(heston_vols[:, -1])):.1%}")
    print()
    
    # Summary comparison
    print("3. MODEL COMPARISON AND CAPITAL REQUIREMENTS")
    print("-" * 50)
    
    comparison_df = pd.DataFrame({
        'Model': ['GBM', 'Heston'],
        'Mean_Unhedged': [results['GBM']['mean_unhedged'], results['Heston']['mean_unhedged']],
        'Std_Unhedged': [results['GBM']['std_unhedged'], results['Heston']['std_unhedged']],
        'VaR_99.5%_Unhedged': [results['GBM']['var_995_unhedged'], results['Heston']['var_995_unhedged']],
        'Mean_Hedged': [results['GBM']['mean_hedged'], results['Heston']['mean_hedged']],
        'Std_Hedged': [results['GBM']['std_hedged'], results['Heston']['std_hedged']],
        'VaR_99.5%_Hedged': [results['GBM']['var_995_hedged'], results['Heston']['var_995_hedged']],
        'Hedging_Effectiveness': [results['GBM']['volatility_reduction'], results['Heston']['volatility_reduction']]
    })
    
    print("Summary Results:")
    print(comparison_df.round(4))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(f'Output/VA_Analysis_Results_{timestamp}.csv', index=False)
    
    # Capital requirement calculation (simplified Solvency II approach)
    print("\n4. SOLVENCY II CAPITAL REQUIREMENTS")
    print("-" * 50)
    
    for model in ['GBM', 'Heston']:
        scr_unhedged = -results[model]['var_995_unhedged']  # 99.5% VaR as SCR proxy
        scr_hedged = -results[model]['var_995_hedged']
        capital_benefit = scr_unhedged - scr_hedged
        
        print(f"{model} Model:")
        print(f"  SCR without hedging: ${scr_unhedged:,.0f}")
        print(f"  SCR with hedging: ${scr_hedged:,.0f}")
        print(f"  Capital benefit from hedging: ${capital_benefit:,.0f} ({capital_benefit/scr_unhedged:.1%})")
        print()
    
    print("="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results saved to Output folder")
    print("="*80)
    
    return results, comparison_df

# Execute the comprehensive analysis
if __name__ == "__main__":
    analysis_results, summary_table = run_comprehensive_analysis()