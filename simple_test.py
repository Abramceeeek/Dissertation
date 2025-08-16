#!/usr/bin/env python3

import sys
import os
sys.path.append('Code')

from Step00_Configuration import *
from Step14_Heston_Simulation import simulate_heston
from Step33_GMAB_Product_Module import GMABParams, evolve_account_from_prices

print("Testing basic simulation components...")

# Use very small parameters
n_paths = 5
N = 50

print(f"Testing with {n_paths} paths and {N} time steps")

try:
    # Test Heston simulation
    print("Testing Heston simulation...")
    S, V = simulate_heston(S0, 0.005, 1.0, N, n_paths, heston_params, seed=seed)
    print(f"Heston simulation successful: S shape {S.shape}, V shape {V.shape}")
    
    # Test account evolution
    print("Testing account evolution...")
    gmab_params = GMABParams()
    dt_years = 1.0 / N
    A = evolve_account_from_prices(S, gmab_params.fee_annual, dt_years)
    print(f"Account evolution successful: A shape {A.shape}")
    
    print("Basic simulation test completed successfully!")
    
except Exception as e:
    print(f"Error in basic simulation: {e}")
    import traceback
    traceback.print_exc()