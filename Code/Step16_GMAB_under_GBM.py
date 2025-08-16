"""
GMAB Variable Annuity Analysis under Geometric Brownian Motion (GBM)

This script implements the GMAB (Guaranteed Minimum Accumulation Benefit) product
under the GBM model for the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Add paths for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration and modules
from Step00_Configuration import *
from Step33_GMAB_Product_Module import GMABParams, evolve_account_from_prices, gmab_value_and_delta, gmab_maturity_payoff
from Step13_GBM_Simulation import simulate_gbm

# Output directories
results_dir = "Output/results"
tables_dir = "Output/tables"
figures_dir = "Output/figures"
results_gbm_file = f"{results_dir}/results_gbm_gmab.csv"
table_gbm_file = f"{tables_dir}/table_gbm.tex"
figure_dpi = 300
figure_format = 'png'

def get_rebalance_times(N, rebalance_freq):
    """Get rebalancing time indices."""
    times = list(range(0, N, rebalance_freq))  # Don't include N to avoid index out of bounds
    if N not in times:
        times.append(N)  # Add final time step if not already included
    return [min(t, N-1) for t in times]  # Ensure no index exceeds N-1

def format_risk_metrics(pnl_array):
    """Calculate and format risk metrics from P&L array."""
    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array)
    var_99_5 = np.percentile(pnl_array, 0.5)  # 99.5% VaR
    cte_99_5 = np.mean(pnl_array[pnl_array <= var_99_5])  # 99.5% CTE
    min_pnl = np.min(pnl_array)
    max_pnl = np.max(pnl_array)
    
    return {
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'var_99_5': var_99_5,
        'cte_99_5': cte_99_5,
        'min_pnl': min_pnl,
        'max_pnl': max_pnl
    }

def load_market_data():
    """Load risk-free rate and dividend yield data."""
    try:
        # Load risk-free rates
        rf_data = pd.read_csv(riskfree_file)
        rf_data['maturity_days'] = rf_data['maturity_days'].astype(int)
        
        # Load dividend yields  
        div_data = pd.read_csv(dividend_file)
        
        # Get rates for T=7 year maturity (approximately 2555 days)
        target_days = int(T * 365.25)
        closest_maturity = rf_data.iloc[(rf_data['maturity_days'] - target_days).abs().argsort()[:1]]
        r = closest_maturity['rate'].iloc[0] / 100  # Convert from percentage
        
        # Use average dividend yield
        q = div_data['dividend_yield'].mean() / 100  # Convert from percentage
        
        return r, q
    except Exception as e:
        print(f"Warning: Could not load market data: {e}")
        print(f"Using default rates: r={r_default}, q={q_default}")
        return r_default, q_default

def run_gmab_simulation():
    """Run GMAB simulation under GBM model with dynamic hedging."""
    
    print("=" * 60)
    print("GMAB Analysis under Geometric Brownian Motion")
    print("=" * 60)
    print(f"Contract Parameters: T={T} years, Guarantee={g_annual:.1%} p.a.")
    print(f"Simulation: {n_paths:,} paths, {int(N):,} time steps")
    print()
    
    # Set random seed
    if use_common_random_numbers:
        np.random.seed(seed)
    
    # Load market data
    r, q = load_market_data()
    print(f"Market rates: r={r:.3%}, q={q:.3%}")
    
    # Initialize GMAB parameters
    gmab_params = GMABParams(
        T_years=T,
        g_annual=g_annual,
        fee_annual=fee_annual,
        trans_cost_bps=trans_cost_bps,
        rebalance_freq=rebalance_freq
    )
    
    # Use actual initial stock price as premium for guarantee calculations
    actual_premium = S0
    
    print(f"GMAB fees: Management={fee_annual:.1%}, Transaction costs={trans_cost_bps} bps")
    print(f"Initial premium: ${actual_premium:,.2f}")
    print()
    
    # Simulate stock price paths using GBM
    print("Simulating GBM paths...")
    mu = r - q  # Risk-neutral drift
    S = simulate_gbm(S0, mu, gbm_params['sigma'], T, int(N), n_paths, seed=seed)
    # S is already in (n_steps+1, n_paths) format from simulate_gbm
    
    print(f"Initial price: ${S[0, 0]:.2f}")
    print(f"Average final price: ${np.mean(S[-1, :]):.2f}")
    print()
    
    # Evolve account values with fees
    print("Calculating account evolution...")
    A = evolve_account_from_prices(S, gmab_params.fee_annual, dt_years)
    
    # Dynamic hedging simulation
    print("Running dynamic hedging simulation...")
    
    # Get rebalancing times
    rebalance_times = get_rebalance_times(int(N), gmab_params.rebalance_freq)
    n_rebalance = len(rebalance_times)
    
    print(f"Array dimensions: S shape {S.shape}, A shape {A.shape}")
    print(f"Rebalancing times: {rebalance_times[:10]}... (total: {n_rebalance})")
    print(f"Max rebalancing index: {max(rebalance_times)}")
    
    # Ensure rebalancing times are within bounds
    max_valid_index = S.shape[0] - 1
    rebalance_times = [min(t, max_valid_index) for t in rebalance_times]
    print(f"Adjusted max rebalancing index: {max(rebalance_times)}")
    
    # Initialize arrays
    hedge_pnl = np.zeros(n_paths)
    unhedged_pnl = np.zeros(n_paths)
    transaction_costs = np.zeros(n_paths)
    
    # Track hedging portfolio for each path
    for path in range(n_paths):
        if path % 1000 == 0:
            print(f"Processing path {path:,} / {n_paths:,}")
        
        path_hedge_pnl = 0.0
        path_trans_cost = 0.0
        prev_delta = 0.0
        
        # Rebalancing loop
        for i, t_idx in enumerate(rebalance_times[:-1]):  # Exclude maturity
            t = t_idx * dt_years
            T_remain = T - t
            
            current_S = S[t_idx, path]
            current_A = A[t_idx, path]
            
            # Calculate GMAB value and delta
            value, delta = gmab_value_and_delta(
                S=current_S,
                A=current_A, 
                T=T_remain,
                r=r,
                q=q,
                gmab_params=gmab_params,
                model='bs'
            )
            
            # Calculate delta change and transaction costs
            delta_change = delta - prev_delta
            trade_notional = abs(delta_change * current_S)
            path_trans_cost += trade_notional * (trans_cost_bps / 10000)
            
            # Update hedge P&L from stock position
            if i > 0:
                prev_S = S[rebalance_times[i-1], path]
                stock_pnl = prev_delta * (current_S - prev_S)
                path_hedge_pnl += stock_pnl
            
            prev_delta = delta
        
        # Final settlement at maturity
        final_S = S[-1, path]
        final_A = A[-1, path]
        
        # Final hedge P&L
        if len(rebalance_times) > 1:
            prev_S = S[rebalance_times[-2], path]  
            stock_pnl = prev_delta * (final_S - prev_S)
            path_hedge_pnl += stock_pnl
        
        # Calculate payoffs
        maturity_payoff = gmab_maturity_payoff(
            final_A, gmab_params, actual_premium
        )
        
        # Debug output for first few paths
        if path < 5:
            G = actual_premium * (1 + gmab_params.g_annual) ** gmab_params.T_years
            print(f"Path {path}: final_A={final_A:.2f}, G={G:.2f}, payoff={maturity_payoff:.6f}")
        
        # Store results
        unhedged_pnl[path] = -maturity_payoff  # Negative for insurer perspective
        hedge_pnl[path] = -maturity_payoff + path_hedge_pnl - path_trans_cost
        transaction_costs[path] = path_trans_cost
    
    print()
    print("Simulation completed. Calculating risk metrics...")
    
    # Calculate risk metrics
    unhedged_metrics = format_risk_metrics(unhedged_pnl)
    hedged_metrics = format_risk_metrics(hedge_pnl)
    
    # Create results summary
    results = {
        'Model': 'GBM',
        'Contract': 'GMAB',
        'Paths': n_paths,
        'T_years': T,
        'Guarantee_Rate': g_annual,
        'Management_Fee': fee_annual,
        'Transaction_Cost_bps': trans_cost_bps,
        'Rebalance_Freq_days': rebalance_freq,
        'Risk_Free_Rate': r,
        'Dividend_Yield': q,
        'GBM_Volatility': gbm_params['sigma'],
        # Unhedged metrics
        'Unhedged_Mean_PnL': unhedged_metrics['mean_pnl'],
        'Unhedged_Std_PnL': unhedged_metrics['std_pnl'], 
        'Unhedged_VaR_99_5': unhedged_metrics['var_99_5'],
        'Unhedged_CTE_99_5': unhedged_metrics['cte_99_5'],
        'Unhedged_Min_PnL': unhedged_metrics['min_pnl'],
        'Unhedged_Max_PnL': unhedged_metrics['max_pnl'],
        # Hedged metrics
        'Hedged_Mean_PnL': hedged_metrics['mean_pnl'],
        'Hedged_Std_PnL': hedged_metrics['std_pnl'],
        'Hedged_VaR_99_5': hedged_metrics['var_99_5'],
        'Hedged_CTE_99_5': hedged_metrics['cte_99_5'], 
        'Hedged_Min_PnL': hedged_metrics['min_pnl'],
        'Hedged_Max_PnL': hedged_metrics['max_pnl'],
        # Additional metrics
        'Avg_Transaction_Cost': np.mean(transaction_costs),
        'Hedge_Effectiveness': 1 - hedged_metrics['std_pnl'] / unhedged_metrics['std_pnl'],
        'Seed': seed
    }
    
    return results, unhedged_pnl, hedge_pnl, S, A

def save_results_and_plots(results, unhedged_pnl, hedge_pnl, S, A):
    """Save simulation results, tables, and plots."""
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True) 
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_gbm_file, index=False)
    print(f"Results saved to: {results_gbm_file}")
    
    # Create LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{GMAB Risk Metrics under GBM Model}}
\\label{{tab:gmab_gbm_results}}
\\begin{{tabular}}{{lrr}}
\\toprule
Metric & Unhedged & Hedged \\\\
\\midrule
Mean P\\&L & {results['Unhedged_Mean_PnL']:.2f} & {results['Hedged_Mean_PnL']:.2f} \\\\
Std Dev P\\&L & {results['Unhedged_Std_PnL']:.2f} & {results['Hedged_Std_PnL']:.2f} \\\\
99.5\\% VaR & {results['Unhedged_VaR_99_5']:.2f} & {results['Hedged_VaR_99_5']:.2f} \\\\
99.5\\% CTE & {results['Unhedged_CTE_99_5']:.2f} & {results['Hedged_CTE_99_5']:.2f} \\\\
Min P\\&L & {results['Unhedged_Min_PnL']:.2f} & {results['Hedged_Min_PnL']:.2f} \\\\
Max P\\&L & {results['Unhedged_Max_PnL']:.2f} & {results['Hedged_Max_PnL']:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(table_gbm_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_gbm_file}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # P&L distributions
    axes[0, 0].hist(unhedged_pnl, bins=50, alpha=0.7, label='Unhedged', density=True)
    axes[0, 0].hist(hedge_pnl, bins=50, alpha=0.7, label='Hedged', density=True)
    axes[0, 0].set_xlabel('P&L')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('GMAB P&L Distributions (GBM)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample stock price paths
    sample_paths = min(10, S.shape[1])
    for i in range(sample_paths):
        axes[0, 1].plot(S[:, i], alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Stock Price')
    axes[0, 1].set_title(f'Sample Stock Price Paths (GBM, {sample_paths} of {n_paths:,})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample account value paths
    for i in range(sample_paths):
        axes[1, 0].plot(A[:, i], alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Account Value')  
    axes[1, 0].set_title(f'Sample Account Value Paths (GMAB, {sample_paths} of {n_paths:,})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Risk metrics comparison
    metrics = ['Mean P&L', 'Std Dev', '99.5% VaR', '99.5% CTE']
    unhedged_values = [results['Unhedged_Mean_PnL'], results['Unhedged_Std_PnL'], 
                      results['Unhedged_VaR_99_5'], results['Unhedged_CTE_99_5']]
    hedged_values = [results['Hedged_Mean_PnL'], results['Hedged_Std_PnL'],
                    results['Hedged_VaR_99_5'], results['Hedged_CTE_99_5']]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, unhedged_values, width, label='Unhedged', alpha=0.8)
    axes[1, 1].bar(x + width/2, hedged_values, width, label='Hedged', alpha=0.8)
    axes[1, 1].set_xlabel('Risk Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Risk Metrics Comparison (GBM)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_file = f"{figures_dir}/gmab_analysis_gbm.png"
    plt.savefig(figure_file, dpi=figure_dpi, format=figure_format, bbox_inches='tight')
    plt.show()
    print(f"Plots saved to: {figure_file}")
    
    # Print summary
    print()
    print("=" * 60)
    print("GMAB GBM SIMULATION RESULTS")
    print("=" * 60)
    print(f"Contract: T={T} years, Guarantee={g_annual:.1%} p.a.")
    print(f"Simulation: {n_paths:,} paths, Seed={seed}")
    print()
    print("Risk Metrics:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Unhedged':<12} {'Hedged':<12}")
    print("-" * 40)
    print(f"{'Mean P&L':<20} {results['Unhedged_Mean_PnL']:<12.2f} {results['Hedged_Mean_PnL']:<12.2f}")
    print(f"{'Std Dev P&L':<20} {results['Unhedged_Std_PnL']:<12.2f} {results['Hedged_Std_PnL']:<12.2f}")
    print(f"{'99.5% VaR':<20} {results['Unhedged_VaR_99_5']:<12.2f} {results['Hedged_VaR_99_5']:<12.2f}")
    print(f"{'99.5% CTE':<20} {results['Unhedged_CTE_99_5']:<12.2f} {results['Hedged_CTE_99_5']:<12.2f}")
    print("-" * 40)
    print(f"Hedge Effectiveness: {results['Hedge_Effectiveness']:.1%}")
    print(f"Avg Transaction Cost: {results['Avg_Transaction_Cost']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    # Run GMAB simulation under GBM
    results, unhedged_pnl, hedge_pnl, S, A = run_gmab_simulation()
    
    # Save results and create plots
    save_results_and_plots(results, unhedged_pnl, hedge_pnl, S, A)
    
    print("\\nGMAB GBM analysis completed successfully!")