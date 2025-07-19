import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import get_r_for_discounting, apply_rila_payoff

# ✅ Load simulated SPX paths from Heston model
paths = pd.read_csv('Output/simulations/SPX_Heston_paths.csv', index_col=0)

# ✅ Parameters for RILA logic
initial_account = 1000
buffer = 0.1   # 10% downside buffer
cap = 0.5      # 50% upside cap

# ✅ Extract start and end values
start_values = paths.iloc[0].values
end_values = paths.iloc[-1].values

# ✅ Calculate returns
returns = (end_values - start_values) / start_values

# ✅ Step 1: Clip returns to avoid extreme outliers
returns = np.clip(returns, -0.9, 1.5)

# ✅ Step 2: Apply RILA payoff logic using utility function
credited_returns = apply_rila_payoff(returns, buffer, cap)

# ✅ Step 3: Calculate final account values
final_accounts = initial_account * (1 + credited_returns)

# ✅ Print sample returns and credited values for debugging
print("Sample returns and credited values:")
for i in range(10):
    print(f"Return: {returns[i]:.4f}, Credited: {credited_returns[i]:.4f}, Final: ${final_accounts[i]:.2f}")

# ✅ Step 4: Temporarily skip discounting for debugging
# r_discount = get_r_for_discounting(
#     '2018-01-03',
#     'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
# )
# T = 7
# discount_factor = np.exp(-r_discount * T)
# discounted_accounts = final_accounts * discount_factor
# print(f"Discount factor: {discount_factor:.8f}")
# print(f"r_discount: {r_discount}")

# ✅ Step 5: Plot the undiscounted distribution
plt.figure(figsize=(10, 6))
plt.hist(final_accounts, bins=100, alpha=0.7, edgecolor='black')
plt.title("Distribution of Final VA Account Values (Heston RILA Logic, Undiscounted)")
plt.xlabel("Account Value")
plt.ylabel("Frequency")
plt.grid(True)

# ✅ Save plot
os.makedirs('Output/plots', exist_ok=True)
plt.savefig('Output/plots/va_distribution_heston_undiscounted.png')
plt.show()

# ✅ Step 6: Summary statistics
print("\nSimulation Summary (Heston RILA, Undiscounted):")
print(f"Final Mean: {final_accounts.mean():,.2f}")
print(f"Final Std: {final_accounts.std():,.2f}")
print(f"5% VaR: {np.percentile(final_accounts, 5):,.2f}")
print(f"1% VaR: {np.percentile(final_accounts, 1):,.2f}")
print(f"Worst Case: {final_accounts.min():,.2f}")
print(f"Best Case: {final_accounts.max():,.2f}")
