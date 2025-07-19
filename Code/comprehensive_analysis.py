import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from hedging_utils import simulate_dynamic_hedge, analyze_hedging_performance
from utils import apply_rila_payoff

def run_complete_analysis():
    """Run comprehensive analysis across all models with hedging."""
    
    print("="*60)
    print("COMPREHENSIVE RILA HEDGING ANALYSIS")
    print("="*60)
    
    # Model configurations
    models = [
        {
            'name': 'Heston',
            'file': 'Output/simulations/SPX_Heston_paths.csv',
            'buffer': 0.1,
            'cap': 0.5,
            'max_paths': 2000  # Limit for performance
        },
        {
            'name': 'GBM', 
            'file': 'Output/simulations/SPX_GBM_paths.csv',
            'buffer': 0.1,
            'cap': 0.12,
            'max_paths': 2000
        },
        {
            'name': 'RoughVol',
            'file': 'Output/simulations/SPX_RoughVol_paths.csv', 
            'buffer': 0.1,
            'cap': 0.5,
            'max_paths': 2000
        }
    ]
    
    results_summary = []
    all_model_results = {}
    
    for model in models:
        print(f"\nAnalyzing {model['name']} Model...")
        print("-" * 40)
        
        try:
            # Load paths
            paths_df = pd.read_csv(model['file'], index_col=0)
            n_paths = min(model['max_paths'], paths_df.shape[1])
            price_paths = paths_df.iloc[:, :n_paths].values
            
            S0 = price_paths[0, 0]
            print(f"  Loaded {n_paths} paths, S0 = ${S0:.2f}")
            
            # Calculate unhedged outcomes
            final_returns = (price_paths[-1, :] - S0) / S0
            credited_returns = apply_rila_payoff(final_returns, model['buffer'], model['cap'])
            unhedged_liability = S0 * (1 + credited_returns)
            unhedged_pnl = S0 - unhedged_liability
            
            # Analyze unhedged risk
            unhedged_stats = analyze_hedging_performance(unhedged_pnl)
            print(f"  Unhedged - Mean P&L: ${unhedged_stats['mean_pnl']:.2f}, Std: ${unhedged_stats['std_pnl']:.2f}")
            
            # Hedging analysis with different frequencies
            hedge_frequencies = {'Daily': 1, 'Weekly': 5, 'Monthly': 21}
            hedge_results = {}
            
            for freq_name, freq_days in hedge_frequencies.items():
                print(f"  Running {freq_name} hedging...")
                
                hedge_pnl, _ = simulate_dynamic_hedge(
                    price_paths, S0, r=0.02, q=0.01, sigma=0.2,
                    buffer=model['buffer'], cap=model['cap'],
                    rebalance_freq=freq_days, transaction_cost=0.001
                )
                
                hedge_stats = analyze_hedging_performance(hedge_pnl, unhedged_pnl)
                hedge_results[freq_name] = hedge_stats
                
                print(f"    {freq_name} - Mean P&L: ${hedge_stats['mean_pnl']:.2f}, " +
                      f"Std: ${hedge_stats['std_pnl']:.2f}, " +
                      f"Risk Reduction: {hedge_stats.get('risk_reduction_std', 0)*100:.1f}%")
            
            # Store results
            model_result = {
                'model': model['name'],
                'unhedged': unhedged_stats,
                'hedged': hedge_results,
                'params': model
            }
            all_model_results[model['name']] = model_result
            
            # Add to summary
            for strategy in ['Unhedged'] + list(hedge_frequencies.keys()):
                if strategy == 'Unhedged':
                    stats = unhedged_stats
                else:
                    stats = hedge_results[strategy]
                
                results_summary.append({
                    'Model': model['name'],
                    'Strategy': strategy,
                    'Mean_PnL': stats['mean_pnl'],
                    'Std_PnL': stats['std_pnl'],
                    'VaR_95': stats['var_95'],
                    'VaR_99': stats['var_99'],
                    'CTE_95': stats['cte_95'],
                    'Worst_Case': stats['worst_case'],
                    'Prob_Loss': stats['prob_loss']
                })
            
        except Exception as e:
            print(f"  Error processing {model['name']}: {e}")
            continue
    
    # Create summary table
    summary_df = pd.DataFrame(results_summary)
    
    # Save results
    os.makedirs('Output', exist_ok=True)
    summary_df.to_csv('Output/comprehensive_hedging_analysis.csv', index=False, float_format='%.2f')
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    print(summary_df.round(2).to_string(index=False))
    
    # Create visualizations
    create_summary_plots(all_model_results)
    
    return summary_df, all_model_results

def create_summary_plots(all_results):
    """Create comprehensive comparison plots."""
    
    # 1. Risk reduction comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(all_results.keys())
    strategies = ['Daily', 'Weekly', 'Monthly']
    
    # VaR comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2
    
    for i, strategy in enumerate(strategies):
        var_95_values = []
        for model in models:
            if model in all_results:
                var_95 = all_results[model]['hedged'][strategy]['var_95']
                var_95_values.append(var_95)
            else:
                var_95_values.append(0)
        
        ax.bar(x + i*width, var_95_values, width, label=f'{strategy} Hedging')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('95% VaR ($)')
    ax.set_title('95% VaR by Model and Hedging Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Standard deviation comparison
    ax = axes[0, 1]
    unhedged_stds = []
    hedged_stds = []
    
    for model in models:
        if model in all_results:
            unhedged_stds.append(all_results[model]['unhedged']['std_pnl'])
            hedged_stds.append(all_results[model]['hedged']['Daily']['std_pnl'])
        else:
            unhedged_stds.append(0)
            hedged_stds.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, unhedged_stds, width, label='Unhedged', alpha=0.7)
    ax.bar(x + width/2, hedged_stds, width, label='Daily Hedged', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('P&L Standard Deviation ($)')
    ax.set_title('Risk Reduction: Unhedged vs Daily Hedged')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Risk reduction percentages
    ax = axes[1, 0]
    risk_reductions = []
    for model in models:
        if model in all_results:
            daily_hedge = all_results[model]['hedged']['Daily']
            risk_red = daily_hedge.get('risk_reduction_std', 0) * 100
            risk_reductions.append(risk_red)
        else:
            risk_reductions.append(0)
    
    bars = ax.bar(models, risk_reductions, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel('Risk Reduction (%)')
    ax.set_title('Risk Reduction from Daily Hedging')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, risk_reductions):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    # Mean P&L comparison
    ax = axes[1, 1]
    mean_pnls = []
    for model in models:
        if model in all_results:
            mean_pnls.append(all_results[model]['hedged']['Daily']['mean_pnl'])
        else:
            mean_pnls.append(0)
    
    bars = ax.bar(models, mean_pnls, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel('Mean P&L ($)')
    ax.set_title('Mean Hedging P&L by Model')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Output/comprehensive_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComprehensive analysis plots saved to Output/comprehensive_analysis_summary.png")

if __name__ == "__main__":
    summary_df, model_results = run_complete_analysis()
    
    print(f"\nAnalysis completed successfully!")
    print(f"Summary saved to: Output/comprehensive_hedging_analysis.csv")
    print(f"Plots saved to: Output/comprehensive_analysis_summary.png")