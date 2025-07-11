import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import get_r_for_discounting

# PARAMETERS
initial_account = 1000
sim_date = '2018-01-03'
term_years = 7
buffer = 0.10     # 10% buffer
cap = 0.12        # 12% cap

# Load GBM simulated paths
paths = pd.read_csv('Output/simulations/SPX_GBM_paths.csv', index_col=0)

print("✅ Loaded GBM paths:", paths.shape)

# Calculate simple returns
start_values = paths.iloc[0].values
end_values = paths.iloc[-1].values
returns = (end_values - start_values) / start_values

print("✅ Example returns:", returns[:5])

# Apply RILA payoff logic
final_accounts = []
for r in returns:
    if r >= 0:
        credited = min(r, cap)
    else:
        if abs(r) <= buffer:
            credited = 0
        else:
            credited = r + buffer
    final_account = initial_account * (1 + credited)
    final_accounts.append(final_account)

final_accounts = np.array(final_accounts)

print(f"✅ Applied RILA logic. Example final accounts: {final_accounts[:5]}")

# Discount to present value
r_discount = get_r_for_discounting(
    sim_date,
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)

discount_factor = np.exp(-r_discount * term_years)
discounted_accounts = final_accounts * discount_factor

print(f"✅ Applied discounting with rate {r_discount:.4f} over {term_years} years.")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(discounted_accounts, bins=75, edgecolor='k', alpha=0.7)
plt.title(f"Distribution of Present Value VA Account Values (GBM + RILA)\nSim Date: {sim_date}")
plt.xlabel("Present Value Account Value")
plt.ylabel("Frequency")
plt.grid(True)

os.makedirs('Output/plots', exist_ok=True)
plt.savefig(f'Output/plots/va_distribution_gbm_rila_{sim_date}.png')
plt.show()

# Summary statistics
print("\n✅ Simulation Summary (GBM + RILA):")
print(f"Final Mean: {discounted_accounts.mean():,.2f}")
print(f"Final Std: {discounted_accounts.std():,.2f}")
print(f"5% VaR: {np.percentile(discounted_accounts, 5):,.2f}")
print(f"1% VaR: {np.percentile(discounted_accounts, 1):,.2f}")
print(f"Worst Case: {discounted_accounts.min():,.2f}")
print(f"Best Case: {discounted_accounts.max():,.2f}")
