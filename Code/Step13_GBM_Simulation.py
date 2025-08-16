import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Step00_Configuration import S0, n_paths, T, N, gbm_params, seed

def simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.normal(0, 1, (n_paths, int(N)))
    
    S = np.zeros((n_paths, int(N) + 1))
    S[:, 0] = S0
    
    for t in range(int(N)):
        S[:, t + 1] = S[:, t] * np.exp(drift + diffusion * Z[:, t])
    
    return S.T

S = simulate_gbm(S0, gbm_params['mu'], gbm_params['sigma'], T, N, n_paths, seed=seed)

import datetime
start_date = datetime.date(2023, 1, 1)
dates = pd.bdate_range(start=start_date, periods=N + 1)
sample_paths = pd.DataFrame(S[:, :10], index=dates)

os.makedirs("Output/simulations", exist_ok=True)
pd.DataFrame(S, index=dates).to_csv("Output/simulations/SPX_GBM_paths.csv")

plt.figure(figsize=(12, 6))
plt.plot(sample_paths)
plt.title("Sample SPX Paths under GBM Model (10 of 10,000)")
plt.xlabel("Date")
plt.ylabel("SPX Level")
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/simulations/SPX_GBM_sample_plot.png")
plt.show()

print("GBM simulation complete. Paths and plot saved.")
