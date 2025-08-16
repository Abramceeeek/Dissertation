#!/usr/bin/env python3

import sys
import os
sys.path.append('Code')

# Import configuration and modify for testing
from Step00_Configuration import *

# Reduce number of paths for testing
n_paths = 100  # Reduced from 10000 for testing

print("Testing Step 17 with reduced paths...")
print(f"Number of paths: {n_paths}")

# Import and run the simulation
from Step17_GMAB_under_Heston import run_gmab_simulation_heston

try:
    results, unhedged_pnl, hedge_pnl, S, A = run_gmab_simulation_heston()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Unhedged PnL - Mean: {results['Unhedged_Mean_PnL']:.2f}, Std: {results['Unhedged_Std_PnL']:.2f}")
    print(f"Hedged PnL - Mean: {results['Hedged_Mean_PnL']:.2f}, Std: {results['Hedged_Std_PnL']:.2f}")
    print(f"VaR 99.5% - Unhedged: {results['Unhedged_VaR_99_5']:.2f}, Hedged: {results['Hedged_VaR_99_5']:.2f}")
    print(f"CTE 99.5% - Unhedged: {results['Unhedged_CTE_99_5']:.2f}, Hedged: {results['Hedged_CTE_99_5']:.2f}")
    print(f"Hedge Effectiveness: {results['Hedge_Effectiveness']:.1%}")
    
    # Check for zeros
    print(f"\nZero check:")
    print(f"Unhedged PnL zeros: {np.sum(unhedged_pnl == 0)} / {len(unhedged_pnl)}")
    print(f"Hedged PnL zeros: {np.sum(hedge_pnl == 0)} / {len(hedge_pnl)}")
    
    print("\nStep 17 test completed successfully!")
    
except Exception as e:
    print(f"Error in Step 17: {e}")
    import traceback
    traceback.print_exc()