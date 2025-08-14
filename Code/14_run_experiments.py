"""
RILA Solvency II SCR Experiment Runner

This script runs the complete experiment grid for RILA products under Solvency II:
- Models: Heston and Rough Volatility
- Rebalancing frequencies: daily, weekly, monthly
- Output: SCR summary CSV and diagnostic plots

Author: Abdurakhmonbek Fayzullaev
Purpose: MSc Dissertation - Solvency II SCR for Equity-Linked Variable Annuities
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add Code directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rila_payoff import rila_pv, rila_greeks
from curves import get_market_curves, create_curve_from_data
from dynamic_hedging import run_dynamic_hedge
from scr_one_year import compute_one_year_scr, compare_scr_across_models, create_scr_diagnostic_plots
from heston_pricing_carr_madan import heston_call_price
from heston_pricing_utils import heston_put_price

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simulation_paths(model: str, data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load pre-simulated paths for the specified model.
    
    Args:
        model (str): 'heston' or 'roughvol'
        data_dir (str): Directory containing simulation files
        
    Returns:
        Tuple of (S_paths, t_grid, metadata)
    """
    
    if model == 'heston':
        file_path = os.path.join(data_dir, 'paths_heston_1y.npz')
    elif model == 'roughvol':
        file_path = os.path.join(data_dir, 'paths_rough_1y.npz')
    else:
        raise ValueError(f"Unknown model: {model}")
    
    if not os.path.exists(file_path):
        logger.warning(f"Simulation file {file_path} not found. Generating synthetic paths.")
        return generate_synthetic_paths(model)
    
    logger.info(f"Loading {model} paths from {file_path}")
    
    try:
        data = np.load(file_path)
        S_paths = data['S_paths']
        t_grid = data['t_grid'] if 't_grid' in data else np.linspace(0, 1, S_paths.shape[1])
        
        # Load metadata if available
        metadata = {}
        if 'params' in data:
            metadata['params'] = data['params'].item()
        
        logger.info(f"Loaded {S_paths.shape[0]} paths with {S_paths.shape[1]} time steps")
        return S_paths, t_grid, metadata
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        logger.info("Falling back to synthetic path generation")
        return generate_synthetic_paths(model)

def generate_synthetic_paths(model: str, n_paths: int = 10000, n_steps: int = 252, 
                           S0: float = 4500.0, T: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic paths when simulation files are not available.
    
    Args:
        model (str): 'heston' or 'roughvol'
        n_paths (int): Number of paths
        n_steps (int): Number of time steps
        S0 (float): Initial asset price
        T (float): Time horizon
        
    Returns:
        Tuple of (S_paths, t_grid, metadata)
    """
    
    logger.warning(f"Generating synthetic {model} paths for fallback")
    
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    
    # Default parameters for synthetic generation
    if model == 'heston':
        params = {
            'v0': 0.04,     # Initial variance
            'kappa': 2.0,   # Mean reversion speed
            'theta': 0.04,  # Long-term variance
            'sigma_v': 0.3, # Vol of vol
            'rho': -0.7     # Correlation
        }
        S_paths = simulate_heston_synthetic(n_paths, n_steps, S0, T, params)
        
    elif model == 'roughvol':
        params = {
            'H': 0.1,       # Hurst parameter
            'nu': 0.3,      # Volatility of volatility
            'rho': -0.7,    # Correlation
            'lambda': 2.0,  # Mean reversion
            'v0': 0.04      # Initial variance
        }
        # Use Heston approximation for rough vol
        S_paths = simulate_heston_synthetic(n_paths, n_steps, S0, T, 
                                          {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 
                                           'sigma_v': 0.4, 'rho': -0.7})
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    metadata = {'params': params, 'synthetic': True}
    return S_paths, t_grid, metadata

def simulate_heston_synthetic(n_paths: int, n_steps: int, S0: float, T: float, 
                            params: Dict) -> np.ndarray:
    """Simple Heston simulation for synthetic paths."""
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    
    S[:, 0] = S0
    v[:, 0] = params['v0']
    
    # Extract parameters
    kappa = params['kappa']
    theta = params['theta'] 
    sigma_v = params['sigma_v']
    rho = params['rho']
    r = 0.02  # Risk-free rate
    
    # Correlated random numbers
    for i in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)
        
        # Evolve variance (with Feller boundary condition)
        v[:, i+1] = np.maximum(
            v[:, i] + kappa * (theta - v[:, i]) * dt + sigma_v * np.sqrt(v[:, i]) * Z2 * sqrt_dt,
            1e-6
        )
        
        # Evolve asset price
        S[:, i+1] = S[:, i] * np.exp(
            (r - 0.5 * v[:, i]) * dt + np.sqrt(v[:, i]) * Z1 * sqrt_dt
        )
    
    return S

def create_heston_option_pricer(params: Dict) -> callable:
    """Create Heston option pricing function."""
    
    def price_option(S0: float, K: float, T: float, r: float, q: float, 
                    option_type: str = 'call', **kwargs) -> float:
        v0 = params['v0']
        kappa = params['kappa']
        theta = params['theta']
        sigma_v = params['sigma_v']
        rho = params['rho']
        
        if option_type.lower() == 'call':
            return heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
        elif option_type.lower() == 'put':
            return heston_put_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
    
    return price_option

def create_roughvol_option_pricer(params: Dict) -> callable:
    """Create Rough Vol option pricing function using Heston approximation."""
    
    # Map rough vol parameters to Heston equivalent
    heston_proxy = {
        'v0': params.get('v0', 0.04),
        'kappa': params.get('lambda', 2.0),
        'theta': params.get('v0', 0.04),
        'sigma_v': params.get('nu', 0.3) * 1.2,  # Scaled for roughness effect
        'rho': params.get('rho', -0.7)
    }
    
    logger.info("Using Heston proxy approximation for Rough Volatility option pricing")
    return create_heston_option_pricer(heston_proxy)

def run_single_experiment(model: str, rebalance: str, S_paths: np.ndarray, 
                         t_grid: np.ndarray, model_params: Dict, 
                         rila_params: Dict, market_curves: Tuple,
                         trans_cost_bps: float = 1.0) -> Dict:
    """
    Run a single experiment for given model and rebalancing frequency.
    
    Args:
        model (str): 'heston' or 'roughvol'
        rebalance (str): 'daily', 'weekly', or 'monthly'
        S_paths (np.ndarray): Asset price paths
        t_grid (np.ndarray): Time grid
        model_params (Dict): Model parameters
        rila_params (Dict): RILA product parameters
        market_curves (Tuple): (r_curve, q_curve)
        trans_cost_bps (float): Transaction cost in bps
        
    Returns:
        Dict: Experiment results
    """
    
    logger.info(f"Running experiment: {model} model with {rebalance} rebalancing")
    
    r_curve, q_curve = market_curves
    
    # Create option pricer
    if model == 'heston':
        option_pricer = create_heston_option_pricer(model_params)
        rila_params_with_model = {**rila_params, 'heston_params': model_params}
    elif model == 'roughvol':
        option_pricer = create_roughvol_option_pricer(model_params)
        rila_params_with_model = {**rila_params, 'roughvol_params': model_params}
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Run dynamic hedging
    try:
        hedge_result = run_dynamic_hedge(
            model=model,
            S_paths=S_paths,
            t_grid=t_grid,
            r_curve=r_curve,
            q_curve=q_curve,
            rila_params=rila_params_with_model,
            rebalance=rebalance,
            trans_cost_bps=trans_cost_bps,
            hedge_instruments={"delta": True, "vega": False}
        )
    except Exception as e:
        logger.error(f"Error in dynamic hedging for {model}-{rebalance}: {e}")
        hedge_result = None
    
    # Compute SCR
    try:
        scr_result = compute_one_year_scr(
            model=f"{model}_{rebalance}",
            S_paths=S_paths,
            t_grid=t_grid,
            r_curve=r_curve,
            q_curve=q_curve,
            rila_params=rila_params_with_model,
            hedging_result=hedge_result,
            option_pricer=option_pricer
        )
    except Exception as e:
        logger.error(f"Error in SCR calculation for {model}-{rebalance}: {e}")
        scr_result = None
    
    # Compile experiment result
    result = {
        'model': model,
        'rebalance': rebalance,
        'hedge_result': hedge_result,
        'scr_result': scr_result,
        'success': hedge_result is not None and scr_result is not None
    }
    
    if result['success']:
        logger.info(f"Completed {model}-{rebalance}: "
                   f"SCR VaR 99.5% = {scr_result['SCR_metrics']['VaR_99.5']:,.0f}")
    
    return result

def create_summary_dataframe(experiment_results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from experiment results."""
    
    summary_data = []
    
    for result in experiment_results:
        if not result['success']:
            continue
            
        scr_metrics = result['scr_result']['SCR_metrics']
        of_analysis = result['scr_result']['OF_analysis']
        hedge_stats = result['hedge_result']['trade_stats'] if result['hedge_result'] else {}
        
        row = {
            'model': result['model'],
            'rebalance': result['rebalance'],
            'VaR_99_5': scr_metrics['VaR_99.5'],
            'CTE_99_5': scr_metrics['CTE_99.5'],
            'mean_dOF': of_analysis['mean_delta_OF'],
            'stdev_dOF': of_analysis['std_delta_OF'],
            'mean_hedge_error': hedge_stats.get('avg_hedge_error', np.nan),
            'TC_bps': hedge_stats.get('avg_transaction_cost', np.nan) / 45.0,  # Normalize to bps
            'n_rebalances': hedge_stats.get('total_rebalances', 0),
            'prob_OF_decrease': of_analysis['prob_OF_decrease']
        }
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_experiment_plots(summary_df: pd.DataFrame, experiment_results: List[Dict], 
                          output_dir: str = 'results/plots'):
    """Create experiment summary plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: SCR comparison by model and rebalancing frequency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # VaR comparison
    pivot_var = summary_df.pivot(index='rebalance', columns='model', values='VaR_99_5')
    pivot_var.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Solvency Capital Requirement (VaR 99.5%)')
    ax1.set_ylabel('SCR (VaR 99.5%)')
    ax1.set_xlabel('Rebalancing Frequency')
    ax1.legend(title='Model')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=0)
    
    # CTE comparison
    pivot_cte = summary_df.pivot(index='rebalance', columns='model', values='CTE_99_5')
    pivot_cte.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Conditional Tail Expectation (CTE 99.5%)')
    ax2.set_ylabel('CTE 99.5%')
    ax2.set_xlabel('Rebalancing Frequency')
    ax2.legend(title='Model')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scr_comparison_by_model_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Own Funds volatility comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_vol = summary_df.pivot(index='rebalance', columns='model', values='stdev_dOF')
    pivot_vol.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Own Funds Volatility by Model and Rebalancing Frequency')
    ax.set_ylabel('Standard Deviation of ΔOF')
    ax.set_xlabel('Rebalancing Frequency')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/own_funds_volatility_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Transaction cost analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN values for TC analysis
    tc_df = summary_df.dropna(subset=['TC_bps'])
    if not tc_df.empty:
        pivot_tc = tc_df.pivot(index='rebalance', columns='model', values='TC_bps')
        pivot_tc.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Transaction Costs by Model and Rebalancing Frequency')
        ax.set_ylabel('Average Transaction Cost (bps)')
        ax.set_xlabel('Rebalancing Frequency')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/transaction_costs_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # Plot 4: Delta OF distributions (if raw data available)
    successful_results = [r for r in experiment_results if r['success']]
    if successful_results:
        n_plots = len(successful_results)
        if n_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            axes = axes.flatten()
        
        for i, result in enumerate(successful_results[:6]):  # Max 6 plots
            if i >= len(axes):
                break
                
            ax = axes[i]
            delta_OF = result['scr_result']['raw_data']['delta_OF']
            
            ax.hist(delta_OF, bins=50, alpha=0.7, density=True)
            ax.axvline(np.percentile(delta_OF, 0.5), color='red', linestyle='--', 
                      label=f'VaR 99.5% = {np.percentile(-delta_OF, 99.5):,.0f}')
            ax.set_xlabel('Change in Own Funds (ΔOF)')
            ax.set_ylabel('Density')
            ax.set_title(f"{result['model'].title()} - {result['rebalance'].title()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(successful_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/delta_of_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Experiment plots saved to {output_dir}")

def save_experiment_config(config: Dict, output_dir: str = 'results'):
    """Save experiment configuration as JSON."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    config_file = os.path.join(output_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_with_timestamp, f, indent=2, default=str)
    
    logger.info(f"Experiment configuration saved to {config_file}")

def main():
    parser = argparse.ArgumentParser(description='Run RILA Solvency II SCR experiments')
    parser.add_argument('--models', nargs='+', choices=['heston', 'roughvol'], 
                       default=['heston', 'roughvol'], help='Models to test')
    parser.add_argument('--rebalance', nargs='+', choices=['daily', 'weekly', 'monthly'],
                       default=['daily', 'weekly', 'monthly'], help='Rebalancing frequencies')
    parser.add_argument('--n_paths', type=int, default=50000, help='Number of Monte Carlo paths')
    parser.add_argument('--tc_bps', type=float, default=1.0, help='Transaction cost in basis points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--date', type=str, default='2021-06-01', help='Market data date')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("Starting RILA Solvency II SCR experiments")
    logger.info(f"Models: {args.models}")
    logger.info(f"Rebalancing frequencies: {args.rebalance}")
    logger.info(f"Monte Carlo paths: {args.n_paths:,}")
    logger.info(f"Transaction cost: {args.tc_bps} bps")
    logger.info(f"Random seed: {args.seed}")
    
    # RILA product parameters
    rila_params = {
        'S0': 4500.0,
        'T': 5.0,       # 5-year product (but we only look at 1-year SCR)
        'cap': 0.25,    # 25% cap
        'buffer': 0.10  # 10% buffer
    }
    
    # Load market curves
    try:
        r_curve, q_curve = get_market_curves(args.date)
        logger.info("Loaded market curves successfully")
    except Exception as e:
        logger.warning(f"Could not load market curves: {e}. Using synthetic curves.")
        r_curve = create_curve_from_data([0.25, 1, 2, 5, 10], [0.02, 0.02, 0.02, 0.02, 0.02])
        q_curve = create_curve_from_data([0.25, 1, 2, 5, 10], [0.015, 0.015, 0.015, 0.015, 0.015])
    
    market_curves = (r_curve, q_curve)
    
    # Run experiments
    experiment_results = []
    
    for model in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model.upper()} model")
        logger.info(f"{'='*60}")
        
        # Load simulation paths
        try:
            S_paths, t_grid, metadata = load_simulation_paths(model, args.data_dir)
            
            # Subsample paths if needed
            if S_paths.shape[0] > args.n_paths:
                indices = np.random.choice(S_paths.shape[0], args.n_paths, replace=False)
                S_paths = S_paths[indices]
                logger.info(f"Subsampled to {args.n_paths:,} paths")
            
            model_params = metadata.get('params', {})
            
            # Run experiments for each rebalancing frequency
            for rebalance in args.rebalance:
                try:
                    result = run_single_experiment(
                        model=model,
                        rebalance=rebalance,
                        S_paths=S_paths,
                        t_grid=t_grid,
                        model_params=model_params,
                        rila_params=rila_params,
                        market_curves=market_curves,
                        trans_cost_bps=args.tc_bps
                    )
                    experiment_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed experiment {model}-{rebalance}: {e}")
                    # Add failed result to maintain structure
                    experiment_results.append({
                        'model': model,
                        'rebalance': rebalance,
                        'success': False,
                        'error': str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to process {model} model: {e}")
            continue
    
    # Create summary
    successful_results = [r for r in experiment_results if r.get('success', False)]
    
    if not successful_results:
        logger.error("No successful experiments. Check logs for errors.")
        return 1
    
    logger.info(f"\nCompleted {len(successful_results)} out of {len(experiment_results)} experiments")
    
    # Create summary DataFrame
    summary_df = create_summary_dataframe(experiment_results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save summary CSV
    summary_file = os.path.join(args.output_dir, 'scr_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Results saved to {summary_file}")
    
    # Print summary table
    logger.info("\nExperiment Results Summary:")
    logger.info(summary_df.to_string(index=False, float_format='%.2f'))
    
    # Create plots
    try:
        create_experiment_plots(summary_df, experiment_results, 
                              os.path.join(args.output_dir, 'plots'))
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
    
    # Save configuration
    experiment_config = {
        'models': args.models,
        'rebalance_frequencies': args.rebalance,
        'n_paths': args.n_paths,
        'transaction_cost_bps': args.tc_bps,
        'seed': args.seed,
        'rila_params': rila_params,
        'market_data_date': args.date,
        'successful_experiments': len(successful_results),
        'total_experiments': len(experiment_results)
    }
    
    save_experiment_config(experiment_config, args.output_dir)
    
    logger.info("\nExperiments completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())