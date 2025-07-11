import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import os
from utils import get_r_minus_q

# Parameters for RILA logic
initial_investment = 1000
cap = 0.12           # 12% annual cap
buffer = 0.10        # 10% downside buffer
term_years = 6       # 6-year contract
participation = 1.0  # Full participation
fee_rate = 0.01      # 1% annual fee

# Simulation settings
np.random.seed(42)
n_paths = 10000
n_days = term_years * 252
dt = 1 / 252
mu = get_r_minus_q(
    '2018-01-03',
    'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv',
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)
sigma = 0.20
S0 = 4000

# Simulate GBM paths for SPX
Z = np.random.standard_normal((n_days, n_paths))
SPX_paths = np.zeros_like(Z)
SPX_paths[0] = S0
for t in range(1, n_days):
    SPX_paths[t] = SPX_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

# Compute annual returns
returns = np.zeros((term_years, n_paths))
for year in range(term_years):
    start = year * 252
    end = (year + 1) * 252
    returns[year] = (SPX_paths[end - 1] - SPX_paths[start]) / SPX_paths[start]

# Apply RILA transformation
account_values = np.full(n_paths, initial_investment, dtype=np.float64)
for year in range(term_years):
    capped = np.minimum(returns[year], cap)
    buffered = np.where(capped >= -buffer, capped, capped + buffer)
    credited = np.where(capped >= -buffer, capped, buffered)
    account_values *= (1 + participation * credited)
    account_values *= (1 - fee_rate)

# Summary stats
summary = {
    "Final Mean": np.mean(account_values),
    "Final Std": np.std(account_values),
    "5% VaR": np.percentile(account_values, 5),
    "1% VaR": np.percentile(account_values, 1),
    "Worst Case": np.min(account_values),
    "Best Case": np.max(account_values)
}
print("\nSimulation Summary:")
for k, v in summary.items():
    print(f"{k}: {v:,.2f}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(account_values, bins=100, color='skyblue', edgecolor='black')
plt.title("Distribution of Final VA Account Values (Advanced RILA Logic)")
plt.xlabel("Account Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig('Output/va_distribution.png')
plt.show()
