import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import get_r_for_discounting
from rila.config import initial_account, buffer_level, cap_level, T, riskfree_file
from rila.payoff import apply_rila_payoff

"""
RILA payoff analysis for Heston-simulated SPX paths using rila package modules.
Loads simulated paths, applies RILA logic, discounts to present value, and plots results.
"""
# Load simulated SPX paths from Heston model
paths = pd.read_csv('Output/simulations/SPX_Heston_paths.csv', index_col=0)

# Extract start and end values
start_values = paths.iloc[0].values
end_values = paths.iloc[-1].values

# Calculate returns
returns = (end_values - start_values) / start_values
returns = np.clip(returns, -0.9, 1.5)

# Apply RILA payoff logic (vectorized)
credited = apply_rila_payoff(returns, buffer_level, cap_level)
final_accounts = initial_account * (1 + credited)

# Discount to present value
r_discount = get_r_for_discounting('2018-01-03', riskfree_file)
discount_factor = np.exp(-r_discount * T)
discounted_accounts = final_accounts * discount_factor

# Plot distributions
plt.figure(figsize=(10, 6))
plt.hist(final_accounts, bins=100, alpha=0.5, label='Undiscounted', edgecolor='black')
plt.hist(discounted_accounts, bins=100, alpha=0.7, label='Discounted', edgecolor='red')
plt.title("Distribution of Final VA Account Values (Heston RILA Logic)")
plt.xlabel("Account Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
os.makedirs('Output/plots', exist_ok=True)
plt.savefig('Output/plots/va_distribution_heston_discounted.png')
plt.show()

# Summary statistics
for label, arr in [('Undiscounted', final_accounts), ('Discounted', discounted_accounts)]:
    print(f"\nSimulation Summary (Heston RILA, {label}):")
    print(f"Final Mean: {arr.mean():,.2f}")
    print(f"Final Std: {arr.std():,.2f}")
    print(f"5% VaR: {np.percentile(arr, 5):,.2f}")
    print(f"1% VaR: {np.percentile(arr, 1):,.2f}")
    print(f"Worst Case: {arr.min():,.2f}")
    print(f"Best Case: {arr.max():,.2f}")
