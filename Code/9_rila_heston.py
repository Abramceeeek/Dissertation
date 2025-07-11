import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import get_r_for_discounting

# Load simulated SPX paths from Heston model
paths = pd.read_csv('Output/simulations/SPX_Heston_paths.csv', index_col=0)

# Parameters for RILA logic
initial_account = 1000
buffer_level = 0.9   # 10% downside buffer
cap_level = 1.5      # 50% upside cap

# Extract start and end values
start_values = paths.iloc[0].values
end_values = paths.iloc[-1].values

# Calculate returns
returns = end_values / start_values

# Apply RILA payoff logic
final_accounts = []
for r in returns:
    if r < buffer_level:
        final = initial_account * buffer_level  # protected by buffer floor
    elif r > cap_level:
        final = initial_account * cap_level     # capped upside
    else:
        final = initial_account * r             # within buffer-cap range
    final_accounts.append(final)


# Get discount rate
r_discount = get_r_for_discounting(
    '2018-01-03',
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)

# Discount to present value
T = 7  # your term in years
discount_factor = np.exp(-r_discount * T)
discounted_accounts = final_accounts * discount_factor


# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(discounted_accounts, bins=100, alpha=0.7, edgecolor='black')
plt.title("Distribution of Final VA Account Values (Heston RILA Logic)")
plt.xlabel("Account Value")
plt.ylabel("Frequency")
plt.grid(True)

# Ensure folder exists
os.makedirs('Output/plots', exist_ok=True)
plt.savefig('Output/plots/va_distribution_heston.png')
plt.show()

# Summary statistics
discounted_accounts = np.array(discounted_accounts)
print("\nSimulation Summary (Heston RILA):")
print(f"Final Mean: {discounted_accounts.mean():,.2f}")
print(f"Final Std: {discounted_accounts.std():,.2f}")
print(f"5% VaR: {np.percentile(discounted_accounts, 5):,.2f}")
print(f"1% VaR: {np.percentile(discounted_accounts, 1):,.2f}")
print(f"Worst Case: {discounted_accounts.min():,.2f}")
print(f"Best Case: {discounted_accounts.max():,.2f}")
