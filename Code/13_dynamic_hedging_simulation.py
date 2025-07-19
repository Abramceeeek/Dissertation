import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rila.hedging import simulate_dynamic_hedge, analyze_hedging_performance
from rila.payoff import apply_rila_payoff
from rila.config import S0, buffer_level, cap_level, n_paths, T

"""
Dynamic hedging simulation for RILA under different models using rila package modules.
Loads simulated paths, runs dynamic hedging, analyzes and plots results.
"""
# Set seed for reproducibility
np.random.seed(42)

def run_hedging_simulation(model_name, paths_file, buffer=buffer_level, cap=cap_level):
    print(f"\nRunning dynamic hedging simulation for {model_name}...")
    # Load simulated paths
    try:
        paths_df = pd.read_csv(paths_file, index_col=0)
        price_paths = paths_df.values  # Shape: (n_steps+1, n_paths)
    except FileNotFoundError:
        print(f"Could not find paths file: {paths_file}")
        return None
    S0_local = price_paths[0, 0]  # Initial price from simulation
    r = 0.02  # Risk-free rate (simplified)
    q = 0.01  # Dividend yield (simplified)
    sigma = 0.2  # Hedging volatility assumption
    # Calculate unhedged liability distribution
    final_returns = (price_paths[-1, :] - S0_local) / S0_local
    credited_returns = apply_rila_payoff(final_returns, buffer, cap)
    unhedged_liability = S0_local * (1 + credited_returns)
    unhedged_pnl = S0_local - unhedged_liability  # P&L from insurer perspective
    # Run hedging simulations with different rebalancing frequencies
    rebalance_frequencies = {
        'Daily': 1,
        'Weekly': 5,
        'Monthly': 21
    }
    results = {
        'model': model_name,
        'unhedged_stats': analyze_hedging_performance(unhedged_pnl),
        'hedged_results': {}
    }
    for freq_name, freq_value in rebalance_frequencies.items():
        print(f"  Simulating {freq_name.lower()} rebalancing...")
        hedge_pnl, hedge_portfolio = simulate_dynamic_hedge(
            price_paths, S0_local, r, q, sigma, buffer, cap, 
            rebalance_freq=freq_value, transaction_cost=0.001
        )
        hedge_stats = analyze_hedging_performance(hedge_pnl, unhedged_pnl)
        results['hedged_results'][freq_name] = {
            'stats': hedge_stats,
            'pnl_distribution': hedge_pnl
        }
        print(f"    Mean P&L: ${hedge_stats['mean_pnl']:.2f}")
        print(f"    P&L Std: ${hedge_stats['std_pnl']:.2f}")
        print(f"    95% VaR: ${hedge_stats['var_95']:.2f}")
    return results

def plot_hedging_results(results, save_dir='Output/plots'):
    model_name = results['model']
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].hist(results['hedged_results']['Daily']['pnl_distribution'], 
                    bins=50, alpha=0.7, label='Unhedged', density=True)
    axes[0, 0].set_title(f'{model_name}: Unhedged vs Hedged P&L')
    axes[0, 0].set_xlabel('P&L ($)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True)
    daily_pnl = results['hedged_results']['Daily']['pnl_distribution']
    axes[0, 1].hist(daily_pnl, bins=50, alpha=0.7, color='green', density=True)
    axes[0, 1].set_title(f'{model_name}: Daily Hedging P&L')
    axes[0, 1].set_xlabel('P&L ($)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True)
    weekly_pnl = results['hedged_results']['Weekly']['pnl_distribution']
    axes[1, 0].hist(weekly_pnl, bins=50, alpha=0.7, color='orange', density=True)
    axes[1, 0].set_title(f'{model_name}: Weekly Hedging P&L')
    axes[1, 0].set_xlabel('P&L ($)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True)
    monthly_pnl = results['hedged_results']['Monthly']['pnl_distribution']
    axes[1, 1].hist(monthly_pnl, bins=50, alpha=0.7, color='red', density=True)
    axes[1, 1].set_title(f'{model_name}: Monthly Hedging P&L')
    axes[1, 1].set_xlabel('P&L ($)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hedging_pnl_distributions_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    metrics = ['var_95', 'var_99', 'cte_95', 'std_pnl']
    freq_names = ['Daily', 'Weekly', 'Monthly']
    x = np.arange(len(metrics))
    width = 0.25
    for i, freq in enumerate(freq_names):
        values = [results['hedged_results'][freq]['stats'][metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=f'{freq} Rebalancing')
    ax.set_xlabel('Risk Metrics')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'{model_name}: Risk Metrics by Rebalancing Frequency')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['95% VaR', '99% VaR', '95% CTE', 'Std Dev'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/risk_metrics_comparison_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(all_results):
    summary_data = []
    for model_results in all_results:
        if model_results is None:
            continue
        model_name = model_results['model']
        unhedged = model_results['unhedged_stats']
        summary_data.append({
            'Model': model_name,
            'Strategy': 'Unhedged',
            'Mean P&L': unhedged['mean_pnl'],
            'Std P&L': unhedged['std_pnl'],
            '95% VaR': unhedged['var_95'],
            '99% VaR': unhedged['var_99'],
            '95% CTE': unhedged['cte_95'],
            'Prob(Loss)': unhedged['prob_loss']
        })
        for freq_name in ['Daily', 'Weekly', 'Monthly']:
            hedged = model_results['hedged_results'][freq_name]['stats']
            summary_data.append({
                'Model': model_name,
                'Strategy': f'{freq_name} Hedge',
                'Mean P&L': hedged['mean_pnl'],
                'Std P&L': hedged['std_pnl'],
                '95% VaR': hedged['var_95'],
                '99% VaR': hedged['var_99'],
                '95% CTE': hedged['cte_95'],
                'Prob(Loss)': hedged['prob_loss']
            })
    summary_df = pd.DataFrame(summary_data)
    os.makedirs('Output', exist_ok=True)
    summary_df.to_csv('Output/hedging_summary_table.csv', index=False, float_format='%.4f')

# Main execution
if __name__ == "__main__":
    print("Starting Dynamic Hedging Analysis for RILA Products")
    print("="*60)
    
    # Model configurations
    models = [
        {
            'name': 'Heston',
            'paths_file': 'Output/simulations/SPX_Heston_paths.csv',
            'buffer': buffer_level,
            'cap': cap_level
        },
        {
            'name': 'GBM',
            'paths_file': 'Output/simulations/SPX_GBM_paths.csv',
            'buffer': buffer_level,
            'cap': 0.12
        },
        {
            'name': 'RoughVol',
            'paths_file': 'Output/simulations/SPX_RoughVol_paths.csv',
            'buffer': buffer_level,
            'cap': cap_level
        }
    ]
    
    # Run hedging simulations for all models
    all_results = []
    for model_config in models:
        results = run_hedging_simulation(
            model_config['name'],
            model_config['paths_file'],
            model_config['buffer'],
            model_config['cap']
        )
        
        if results is not None:
            # Plot results for this model
            plot_hedging_results(results)
            all_results.append(results)
        
        print(f"Completed {model_config['name']} analysis")
    
    # Create comprehensive summary
    if all_results:
        summary_df = create_summary_table(all_results)
        print(f"\nAnalysis complete! Summary saved to Output/hedging_summary_table.csv")
        print(f"Generated {len(all_results)} model comparisons with hedging analysis")
    else:
        print("No successful simulations found. Please check that simulation files exist.")