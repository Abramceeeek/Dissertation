import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import get_r_minus_q

# Set seed for reproducibility
np.random.seed(42)

# Heston model parameters
S0 = 4500            # initial SPX level
V0 = 0.04            # initial variance
mu = get_r_minus_q(
    '2018-01-03',
    'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv',
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)                    # drift
kappa = 2.0          # mean reversion speed
theta = 0.04         # long-run variance
sigma_v = 0.3        # vol of vol
rho = -0.7           # correlation between W1 and W2
T = 7                # time horizon in years
N = 252 * T          # number of steps
dt = T / N
n_paths = 10000

# Arrays to store results
S = np.zeros((N + 1, n_paths))
V = np.zeros((N + 1, n_paths))
S[0] = S0
V[0] = V0

# Generate correlated Brownian motions
Z1 = np.random.normal(size=(N, n_paths))
Z2 = np.random.normal(size=(N, n_paths))
W1 = Z1
W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

for t in range(1, N + 1):
    # Variance must stay positive
    V[t] = np.abs(
        V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_v * np.sqrt(V[t-1]) * np.sqrt(dt) * W2[t-1]
    )
    
    S[t] = S[t-1] * np.exp((mu - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1]) * np.sqrt(dt) * W1[t-1])

# Create a DataFrame with a few sample paths for plotting
dates = pd.date_range(start="2023-01-01", periods=N + 1, freq="B")
sample_paths = pd.DataFrame(S[:, :10], index=dates)

# Save the full path data for RILA simulation
os.makedirs("Output/simulations", exist_ok=True)
pd.DataFrame(S, index=dates).to_csv("Output/simulations/SPX_Heston_paths.csv")

# Plot sample paths
plt.figure(figsize=(12, 6))
plt.plot(sample_paths)
plt.title("Sample SPX Paths under Heston Model (10 of 10,000)")
plt.xlabel("Date")
plt.ylabel("SPX Level")
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/simulations/SPX_Heston_sample_plot.png")
plt.show()

print("✅ Heston simulation complete. Paths and plot saved.")
