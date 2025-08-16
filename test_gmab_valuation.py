#!/usr/bin/env python3

import sys
import os
sys.path.append('Code')

from Step33_GMAB_Product_Module import GMABParams, gmab_value_and_delta, gmab_maturity_payoff
from Step00_Configuration import heston_params

print("Testing GMAB valuation function...")

# Test parameters
S = 4500
A = 4500
T = 1.0
r = 0.02
q = 0.015

gmab_params = GMABParams()

print(f"S: {S}, A: {A}, T: {T}, r: {r}, q: {q}")
print(f"GMAB params: T_years={gmab_params.T_years}, g_annual={gmab_params.g_annual}")

try:
    # Test Black-Scholes valuation
    print("\nTesting Black-Scholes valuation...")
    value_bs, delta_bs = gmab_value_and_delta(
        S=S, A=A, T=T, r=r, q=q, 
        gmab_params=gmab_params, model='bs'
    )
    print(f"BS - Value: {value_bs:.2f}, Delta: {delta_bs:.4f}")
    
    # Test Heston valuation
    print("\nTesting Heston valuation...")
    value_heston, delta_heston = gmab_value_and_delta(
        S=S, A=A, T=T, r=r, q=q, 
        gmab_params=gmab_params, model='heston', heston_params=heston_params
    )
    print(f"Heston - Value: {value_heston:.2f}, Delta: {delta_heston:.4f}")
    
    # Test maturity payoff
    print("\nTesting maturity payoff...")
    payoff = gmab_maturity_payoff(A, gmab_params, 1.0)
    print(f"Maturity payoff: {payoff:.2f}")
    
    print("\nGMAB valuation test completed successfully!")
    
except Exception as e:
    print(f"Error in GMAB valuation: {e}")
    import traceback
    traceback.print_exc()