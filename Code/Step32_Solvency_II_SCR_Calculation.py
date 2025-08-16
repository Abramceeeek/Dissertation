"""
Solvency II One-Year SCR Calculation for RILA Products

This module computes the Solvency Capital Requirement (SCR) over one year 
using the change in Own Funds (ΔOF) under stochastic volatility models.

SCR = VaR 99.5% and CTE 99.5% of -ΔOF where:
  ΔOF = (Assets_1y - BEL_1y) - (Assets_0 - BEL_0)

Author: Abdurakhmonbek Fayzullaev
Purpose: MSc Dissertation - Solvency II SCR for Equity-Linked Variable Annuities
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Callable, Optional
from scipy import stats
import logging
from Step23_RILA_Payoff_and_Replication import rila_pv, rila_payoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_var_cte(losses: np.ndarray, confidence_levels: list = [0.995]) -> Dict[str, float]:
    """
    Calculate Value-at-Risk (VaR) and Conditional Tail Expectation (CTE) for given loss distribution.
    
    Args:
        losses (np.ndarray): Loss distribution (positive values = losses)
        confidence_levels (list): Confidence levels (e.g., [0.995] for 99.5%)
        
    Returns:
        Dict containing VaR and CTE at specified confidence levels
    """
    results = {}
    
    for alpha in confidence_levels:
        # VaR: alpha-quantile of loss distribution
        var_alpha = np.percentile(losses, alpha * 100)
        
        # CTE (Expected Shortfall): conditional expectation of losses exceeding VaR
        tail_losses = losses[losses >= var_alpha]
        cte_alpha = np.mean(tail_losses) if len(tail_losses) > 0 else var_alpha
        
        alpha_pct = int(alpha * 1000) / 10  # Convert to percentage (99.5%)
        results[f'VaR_{alpha_pct}'] = var_alpha
        results[f'CTE_{alpha_pct}'] = cte_alpha
    
    # Additional statistics
    results['mean'] = np.mean(losses)
    results['std'] = np.std(losses)
    results['min'] = np.min(losses)
    results['max'] = np.max(losses)
    
    return results

def calculate_bel(S: np.ndarray, T: float, r_curve: Union[Callable, float], 
                 q_curve: Union[Callable, float], rila_params: Dict, 
                 option_pricer: Callable) -> np.ndarray:
    """
    Calculate Best Estimate Liability (BEL) for RILA contracts.
    
    Args:
        S (np.ndarray): Current asset prices (can be array for multiple scenarios)
        T (float): Remaining time to maturity
        r_curve: Risk-free rate curve
        q_curve: Dividend yield curve  
        rila_params (Dict): RILA product parameters
        option_pricer (Callable): Option pricing function
        
    Returns:
        np.ndarray: BEL values corresponding to each S value
    """
    S = np.atleast_1d(S)
    bel_values = np.zeros_like(S)
    
    # Get rate values
    r = r_curve(T) if callable(r_curve) else r_curve
    q = q_curve(T) if callable(q_curve) else q_curve
    
    cap = rila_params['cap']
    buffer = rila_params['buffer']
    
    # Calculate BEL for each scenario
    for i, S_val in enumerate(S):
        if T > 0:
            # Present value using option replication
            bel_values[i] = rila_pv(
                S0=S_val, T=T, r=r, q=q, cap=cap, buffer=buffer,
                option_pricer=option_pricer
            )
        else:
            # At maturity: BEL = terminal payoff
            S0_original = rila_params['S0']  # Original initial price
            bel_values[i] = rila_payoff(np.array([S_val]), S0_original, cap, buffer)[0]
    
    return bel_values

def calculate_assets(hedging_result: Optional[Dict], S_paths: np.ndarray, 
                    t_idx: int) -> np.ndarray:
    """
    Calculate asset values at time t including hedge portfolio.
    
    Args:
        hedging_result (Dict or None): Results from dynamic hedging
        S_paths (np.ndarray): Asset price paths
        t_idx (int): Time index
        
    Returns:
        np.ndarray: Asset values for each path
    """
    S_current = S_paths[:, t_idx]
    
    if hedging_result is None:
        # No hedging: assets = 0 (pure liability holder)
        return np.zeros_like(S_current)
    
    # With hedging: assets = hedge portfolio value
    # This would be calculated from the hedge positions and current market values
    # For now, approximate using hedge P&L up to this point
    n_paths = len(S_current)
    
    # Simplified calculation: assume linear accumulation of hedge P&L
    if t_idx < len(hedging_result.get('pathwise', {}).get('portfolio_values', [[]])[0]):
        # Use pathwise portfolio values if available
        portfolio_values = hedging_result['pathwise']['portfolio_values'][:, min(t_idx, portfolio_values.shape[1]-1)]
        return portfolio_values
    else:
        # Fallback: distribute final hedge P&L proportionally
        time_fraction = t_idx / (S_paths.shape[1] - 1) if S_paths.shape[1] > 1 else 1.0
        return hedging_result['hedge_pnl'] * time_fraction

def compute_one_year_scr(
    model: str,
    S_paths: np.ndarray,
    t_grid: np.ndarray,
    r_curve: Union[Callable, float],
    q_curve: Union[Callable, float],
    rila_params: Dict,
    hedging_result: Optional[Dict] = None,
    option_pricer: Callable = None
) -> Dict:
    """
    Compute Solvency II one-year SCR using change in Own Funds (ΔOF).
    
    Own Funds (OF) = Assets - Best Estimate Liability (BEL)
    ΔOF = OF_1y - OF_0
    SCR = VaR 99.5% and CTE 99.5% of (-ΔOF)
    
    Args:
        model (str): Model name for logging
        S_paths (np.ndarray): Simulated price paths (n_paths, n_steps+1)
        t_grid (np.ndarray): Time grid in years
        r_curve: Risk-free rate curve
        q_curve: Dividend yield curve
        rila_params (Dict): RILA parameters
        hedging_result (Dict, optional): Dynamic hedging results
        option_pricer (Callable): Option pricing function
        
    Returns:
        Dict: SCR metrics and analysis
    """
    
    logger.info(f"Computing one-year SCR for {model} model")
    
    n_paths = S_paths.shape[0]
    
    # Find 1-year time index
    one_year_idx = np.argmin(np.abs(t_grid - 1.0))
    actual_horizon = t_grid[one_year_idx]
    
    logger.info(f"Using time index {one_year_idx} for {actual_horizon:.3f} years")
    
    # Initial conditions (t=0)
    S0 = S_paths[:, 0]
    T0 = rila_params['T']  # Initial time to maturity
    
    # Calculate initial BEL
    logger.info("Calculating initial BEL...")
    BEL_0 = calculate_bel(
        S=S0, T=T0, r_curve=r_curve, q_curve=q_curve,
        rila_params=rila_params, option_pricer=option_pricer
    )
    
    # Calculate initial Assets
    Assets_0 = calculate_assets(hedging_result, S_paths, 0)
    
    # Initial Own Funds
    OF_0 = Assets_0 - BEL_0
    
    # One-year conditions (t=1)
    S1 = S_paths[:, one_year_idx]
    T1 = max(0, T0 - actual_horizon)  # Remaining time to maturity
    
    # Calculate BEL at t=1
    logger.info("Calculating BEL at t=1...")
    BEL_1 = calculate_bel(
        S=S1, T=T1, r_curve=r_curve, q_curve=q_curve,
        rila_params=rila_params, option_pricer=option_pricer
    )
    
    # Calculate Assets at t=1
    Assets_1 = calculate_assets(hedging_result, S_paths, one_year_idx)
    
    # Own Funds at t=1
    OF_1 = Assets_1 - BEL_1
    
    # Change in Own Funds
    delta_OF = OF_1 - OF_0
    
    # SCR calculation: VaR and CTE of negative ΔOF (i.e., losses)
    scr_losses = -delta_OF  # Convert to loss convention (positive = bad)
    
    # Calculate risk metrics
    scr_metrics = calculate_var_cte(scr_losses, confidence_levels=[0.995, 0.99, 0.95])
    
    # Additional Own Funds analysis
    of_analysis = {
        'mean_OF_0': np.mean(OF_0),
        'mean_OF_1': np.mean(OF_1),
        'mean_delta_OF': np.mean(delta_OF),
        'std_delta_OF': np.std(delta_OF),
        'min_delta_OF': np.min(delta_OF),
        'max_delta_OF': np.max(delta_OF),
        'prob_OF_decrease': np.mean(delta_OF < 0),
    }
    
    # Component analysis
    component_analysis = {
        'mean_BEL_change': np.mean(BEL_1 - BEL_0),
        'mean_Assets_change': np.mean(Assets_1 - Assets_0),
        'std_BEL_change': np.std(BEL_1 - BEL_0),
        'std_Assets_change': np.std(Assets_1 - Assets_0),
    }
    
    # Hedging effectiveness (if applicable)
    hedging_effectiveness = {}
    if hedging_result is not None:
        # Compare hedged vs unhedged SCR
        # For unhedged case: Assets = 0, so ΔOF = -ΔBEL
        unhedged_delta_OF = -(BEL_1 - BEL_0)
        unhedged_scr_losses = -unhedged_delta_OF
        unhedged_scr = calculate_var_cte(unhedged_scr_losses, confidence_levels=[0.995])
        
        hedging_effectiveness = {
            'unhedged_SCR_99_5': unhedged_scr['VaR_99.5'],
            'hedged_SCR_99_5': scr_metrics['VaR_99.5'],
            'SCR_reduction': unhedged_scr['VaR_99.5'] - scr_metrics['VaR_99.5'],
            'SCR_reduction_pct': (unhedged_scr['VaR_99.5'] - scr_metrics['VaR_99.5']) / unhedged_scr['VaR_99.5'] * 100,
            'unhedged_vol': np.std(unhedged_scr_losses),
            'hedged_vol': np.std(scr_losses),
            'volatility_reduction_pct': (np.std(unhedged_scr_losses) - np.std(scr_losses)) / np.std(unhedged_scr_losses) * 100
        }
    
    # Compile results
    results = {
        'model': model,
        'time_horizon': actual_horizon,
        'n_paths': n_paths,
        'SCR_metrics': scr_metrics,
        'OF_analysis': of_analysis,
        'component_analysis': component_analysis,
        'hedging_effectiveness': hedging_effectiveness,
        'raw_data': {
            'delta_OF': delta_OF,
            'SCR_losses': scr_losses,
            'OF_0': OF_0,
            'OF_1': OF_1,
            'BEL_0': BEL_0,
            'BEL_1': BEL_1,
            'Assets_0': Assets_0,
            'Assets_1': Assets_1
        }
    }
    
    # Log key results
    logger.info(f"SCR Calculation Results for {model}:")
    logger.info(f"  SCR (VaR 99.5%): {scr_metrics['VaR_99.5']:,.2f}")
    logger.info(f"  SCR (CTE 99.5%): {scr_metrics['CTE_99.5']:,.2f}")
    logger.info(f"  Mean ΔOF: {of_analysis['mean_delta_OF']:,.2f}")
    logger.info(f"  Std ΔOF: {of_analysis['std_delta_OF']:,.2f}")
    logger.info(f"  Prob(OF decrease): {of_analysis['prob_OF_decrease']:.1%}")
    
    if hedging_result is not None:
        logger.info(f"  SCR reduction from hedging: {hedging_effectiveness['SCR_reduction']:,.2f} ({hedging_effectiveness['SCR_reduction_pct']:.1f}%)")
        logger.info(f"  Volatility reduction: {hedging_effectiveness['volatility_reduction_pct']:.1f}%")
    
    return results

def compare_scr_across_models(scr_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare SCR results across different models and create summary table.
    
    Args:
        scr_results (Dict): Dictionary of SCR results by model name
        
    Returns:
        pd.DataFrame: Comparison table
    """
    
    comparison_data = []
    
    for model_name, results in scr_results.items():
        scr_metrics = results['SCR_metrics']
        of_analysis = results['OF_analysis']
        hedging_eff = results.get('hedging_effectiveness', {})
        
        row = {
            'Model': model_name,
            'SCR_VaR_99_5': scr_metrics['VaR_99.5'],
            'SCR_CTE_99_5': scr_metrics['CTE_99.5'],
            'Mean_Delta_OF': of_analysis['mean_delta_OF'],
            'Std_Delta_OF': of_analysis['std_delta_OF'],
            'Prob_OF_Decrease': of_analysis['prob_OF_decrease'],
            'Min_Delta_OF': of_analysis['min_delta_OF'],
            'Max_Delta_OF': of_analysis['max_delta_OF']
        }
        
        # Add hedging effectiveness metrics if available
        if hedging_eff:
            row.update({
                'Unhedged_SCR': hedging_eff.get('unhedged_SCR_99_5', np.nan),
                'SCR_Reduction': hedging_eff.get('SCR_reduction', np.nan),
                'SCR_Reduction_Pct': hedging_eff.get('SCR_reduction_pct', np.nan),
                'Vol_Reduction_Pct': hedging_eff.get('volatility_reduction_pct', np.nan)
            })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_scr_diagnostic_plots(scr_results: Dict[str, Dict], output_dir: str = 'results/plots'):
    """
    Create diagnostic plots for SCR analysis.
    
    Args:
        scr_results (Dict): SCR results by model
        output_dir (str): Output directory for plots
    """
    
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Delta OF distributions
    plt.figure(figsize=(12, 8))
    
    for i, (model, results) in enumerate(scr_results.items()):
        delta_OF = results['raw_data']['delta_OF']
        
        plt.subplot(2, 2, i+1)
        plt.hist(delta_OF, bins=50, alpha=0.7, density=True, label=f'{model}')
        plt.axvline(np.percentile(delta_OF, 0.5), color='red', linestyle='--', 
                   label=f'VaR 99.5% = {np.percentile(-delta_OF, 99.5):.0f}')
        plt.xlabel('Change in Own Funds (ΔOF)')
        plt.ylabel('Density')
        plt.title(f'{model} Model: ΔOF Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scr_delta_of_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: SCR comparison bar chart
    comparison_df = compare_scr_across_models(scr_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # VaR comparison
    ax1.bar(comparison_df['Model'], comparison_df['SCR_VaR_99_5'])
    ax1.set_ylabel('SCR (VaR 99.5%)')
    ax1.set_title('Solvency Capital Requirement Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Volatility comparison
    ax2.bar(comparison_df['Model'], comparison_df['Std_Delta_OF'])
    ax2.set_ylabel('Standard Deviation of ΔOF')
    ax2.set_title('Own Funds Volatility Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scr_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SCR diagnostic plots saved to {output_dir}")

# Example usage and testing
if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Mock parameters
    rila_params = {
        'S0': 4500.0,
        'T': 5.0,  # 5-year product
        'cap': 0.25,
        'buffer': 0.10
    }
    
    # Generate synthetic paths
    n_paths, n_steps = 5000, 252
    t_grid = np.linspace(0, 1, n_steps + 1)
    
    # Simple GBM for testing
    dt = 1 / n_steps
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = rila_params['S0']
    
    for i in range(n_steps):
        S_paths[:, i+1] = S_paths[:, i] * np.exp(0.05 * dt + 0.2 * dW[:, i])
    
    # Mock option pricer
    def mock_option_pricer(S0, K, T, r, q, option_type='call', **kwargs):
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r-q+0.2**2/2)*T) / (0.2*np.sqrt(T)) if T > 0 else 0
        d2 = d1 - 0.2*np.sqrt(T) if T > 0 else 0
        
        if option_type == 'call':
            return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if T > 0 else max(S0-K, 0)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1) if T > 0 else max(K-S0, 0)
    
    # Calculate SCR
    scr_result = compute_one_year_scr(
        model='test_gbm',
        S_paths=S_paths,
        t_grid=t_grid,
        r_curve=0.02,
        q_curve=0.01,
        rila_params=rila_params,
        hedging_result=None,
        option_pricer=mock_option_pricer
    )
    
    print(f"\nTest SCR Results:")
    print(f"SCR (VaR 99.5%): ${scr_result['SCR_metrics']['VaR_99.5']:,.0f}")
    print(f"SCR (CTE 99.5%): ${scr_result['SCR_metrics']['CTE_99.5']:,.0f}")
    print(f"Mean ΔOF: ${scr_result['OF_analysis']['mean_delta_OF']:,.0f}")
    print(f"Probability of OF decrease: {scr_result['OF_analysis']['prob_OF_decrease']:.1%}")