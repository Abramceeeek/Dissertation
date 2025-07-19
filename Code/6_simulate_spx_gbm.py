import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rila.models import simulate_gbm
from rila.config import S0, n_paths, T, N

"""
Simulate SPX paths under the Geometric Brownian Motion (GBM) model using rila package modules.
Saves all paths to CSV and plots a sample.
"""
# Set seed for reproducibility
seed = 42

# GBM parameters
mu = 0.01  # risk-neutral drift (can be replaced with get_r_minus_q)
sigma = 0.2  # annualized volatility

# Simulate paths
S = simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=seed)

# Create a DataFrame with a few sample paths for plotting
import datetime
start_date = datetime.date(2023, 1, 1)
dates = pd.bdate_range(start=start_date, periods=N + 1)
sample_paths = pd.DataFrame(S[:, :10], index=dates)

# Save the full path data for RILA simulation
os.makedirs("Output/simulations", exist_ok=True)
pd.DataFrame(S, index=dates).to_csv("Output/simulations/SPX_GBM_paths.csv")

# Plot sample paths
plt.figure(figsize=(12, 6))
plt.plot(sample_paths)
plt.title("Sample SPX Paths under GBM Model (10 of 10,000)")
plt.xlabel("Date")
plt.ylabel("SPX Level")
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/simulations/SPX_GBM_sample_plot.png")
plt.show()

print("✅ GBM simulation complete. Paths and plot saved.")
