#!/usr/bin/env python3

import sys
import os
sys.path.append('Code')

from Step23_Heston_Carr_Madan_Pricing import carr_madan_call_price
from Step00_Configuration import heston_params

print("Testing Heston pricing function...")

# Test parameters
S0 = 4500
K = 4500
T = 1.0
r = 0.02
q = 0.015

print(f"S0: {S0}, K: {K}, T: {T}, r: {r}, q: {q}")
print(f"Heston params: {heston_params}")

try:
    price = carr_madan_call_price(
        S0, K, T, r, q,
        heston_params['v0'], heston_params['kappa'], heston_params['theta'],
        heston_params['sigma_v'], heston_params['rho']
    )
    print(f"Heston call price: {price:.2f}")
    
    # Test a few more cases
    for K in [4000, 4500, 5000]:
        price = carr_madan_call_price(
            S0, K, T, r, q,
            heston_params['v0'], heston_params['kappa'], heston_params['theta'],
            heston_params['sigma_v'], heston_params['rho']
        )
        print(f"K={K}: price={price:.2f}")
    
    print("Heston pricing test completed successfully!")
    
except Exception as e:
    print(f"Error in Heston pricing: {e}")
    import traceback
    traceback.print_exc()