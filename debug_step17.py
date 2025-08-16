#!/usr/bin/env python3

import sys
import os
sys.path.append('Code')

# Import configuration and modify for testing
from Step00_Configuration import *

# Reduce parameters for debugging
n_paths = 10  # Very small number for debugging
N = 252  # Only 1 year of daily steps

print("Debugging Step 17 with minimal parameters...")
print(f"Number of paths: {n_paths}")
print(f"Number of time steps: {N}")

# Import the simulation function
from Step17_GMAB_under_Heston import run_gmab_simulation_heston

try:
    print("Starting simulation...")
    results, unhedged_pnl, hedge_pnl, S, A = run_gmab_simulation_heston()
    
    print("\n" + "="*60)
    print("DEBUG RESULTS")
    print("="*60)
    print(f"Unhedged PnL - Mean: {results['Unhedged_Mean_PnL']:.2f}, Std: {results['Unhedged_Std_PnL']:.2f}")
    print(f"Hedged PnL - Mean: {results['Hedged_Mean_PnL']:.2f}, Std: {results['Hedged_Std_PnL']:.2f}")
    
    # Check for zeros
    print(f"\nZero check:")
    print(f"Unhedged PnL zeros: {np.sum(unhedged_pnl == 0)} / {len(unhedged_pnl)}")
    print(f"Hedged PnL zeros: {np.sum(hedge_pnl == 0)} / {len(hedge_pnl)}")
    
    # Check for extreme values
    print(f"\nExtreme value check:")
    print(f"Unhedged PnL min: {np.min(unhedged_pnl):.2f}, max: {np.max(unhedged_pnl):.2f}")
    print(f"Hedged PnL min: {np.min(hedge_pnl):.2f}, max: {np.max(hedge_pnl):.2f}")
    
    print("\nDebug completed!")
    
except Exception as e:
    print(f"Error in Step 17: {e}")
    import traceback
    traceback.print_exc()