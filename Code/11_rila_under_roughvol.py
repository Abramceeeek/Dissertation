import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import get_r_for_discounting, apply_rila_payoff

# Load simulated SPX paths under rough volatility
spx_paths = pd.read_csv('Output/simulations/SPX_RoughVol_paths.csv', index_col=0)

# Parameters for the RILA product
initial_investment = 1000
term_years = 7
buffer = 0.10        # 10% downside buffer
cap = 0.50           # 50% upside cap

# Extract start and end values
start_values = spx_paths.iloc[0].values
end_values = spx_paths.iloc[-1].values

# Calculate returns
returns = (end_values - start_values) / start_values

# Apply RILA payoff logic using utility function
credited_returns = apply_rila_payoff(returns, buffer, cap)

# Calculate final account values
final_values = initial_investment * (1 + credited_returns)

# Get discount rate
r_discount = get_r_for_discounting(
    '2018-01-03',
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)

# Discount to present value
T = 7  # your term in years
discount_factor = np.exp(-r_discount * T)
discounted_accounts = final_values * discount_factor

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(discounted_accounts, bins=75, edgecolor='k', alpha=0.7)
plt.title('Distribution of Final VA Account Values (Rough Vol RILA Logic)')
plt.xlabel('Account Value')
plt.ylabel('Frequency')
os.makedirs('Output/surfaces', exist_ok=True)
plt.savefig('Output/surfaces/va_distribution_roughvol.png')
plt.show()

# Print summary
discounted_accounts = np.array(discounted_accounts)
print("Simulation Summary (Rough Vol):")
print(f"Final Mean: {discounted_accounts.mean():,.2f}")
print(f"Final Std: {discounted_accounts.std():,.2f}")
print(f"5% VaR: {np.percentile(discounted_accounts, 5):,.2f}")
print(f"1% VaR: {np.percentile(discounted_accounts, 1):,.2f}")
print(f"Worst Case: {discounted_accounts.min():,.2f}")
print(f"Best Case: {discounted_accounts.max():,.2f}")
