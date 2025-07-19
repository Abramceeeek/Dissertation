import numpy as np
import pandas as pd
from hedging_utils import simulate_dynamic_hedge, analyze_hedging_performance
from utils import apply_rila_payoff

print("Testing dynamic hedging simulation...")

# Load Heston paths (use a subset for testing)
try:
    paths_df = pd.read_csv('Output/simulations/SPX_Heston_paths.csv', index_col=0)
    # Use only first 1000 paths for testing
    price_paths = paths_df.iloc[:, :1000].values  
    print(f"Loaded paths: {price_paths.shape}")
    
    # Parameters
    S0 = price_paths[0, 0]
    r = 0.02
    q = 0.01  
    sigma = 0.2
    buffer = 0.1
    cap = 0.5
    
    print(f"Initial S0: {S0:.2f}")
    print(f"Running hedging simulation...")
    
    # Calculate unhedged liability
    final_returns = (price_paths[-1, :] - S0) / S0
    credited_returns = apply_rila_payoff(final_returns, buffer, cap)
    unhedged_liability = S0 * (1 + credited_returns)
    unhedged_pnl = S0 - unhedged_liability
    
    print(f"Unhedged P&L stats:")
    print(f"  Mean: ${np.mean(unhedged_pnl):.2f}")
    print(f"  Std: ${np.std(unhedged_pnl):.2f}")
    print(f"  95% VaR: ${np.percentile(unhedged_pnl, 5):.2f}")
    
    # Run daily hedging simulation
    print("Running daily hedging simulation...")
    hedge_pnl, hedge_portfolio = simulate_dynamic_hedge(
        price_paths, S0, r, q, sigma, buffer, cap, 
        rebalance_freq=1, transaction_cost=0.001
    )
    
    # Analyze results
    hedge_stats = analyze_hedging_performance(hedge_pnl, unhedged_pnl)
    
    print(f"\nHedging Results:")
    print(f"  Mean P&L: ${hedge_stats['mean_pnl']:.2f}")
    print(f"  P&L Std: ${hedge_stats['std_pnl']:.2f}")
    print(f"  95% VaR: ${hedge_stats['var_95']:.2f}")
    print(f"  Risk Reduction (Std): {hedge_stats.get('risk_reduction_std', 0)*100:.1f}%")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()