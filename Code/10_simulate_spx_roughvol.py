import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import pandas as pd
import os

from utils import get_r_minus_q  # ✅ NEW import

# Set seed for reproducibility
np.random.seed(42)

# Choose your simulation date
sim_date = '2018-01-03'

# Get (r - q) for that date
mu = get_r_minus_q(
    sim_date,
    'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv',
    'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv'
)

# Parameters
n_paths = 10000
n_steps = 252 * 7  # 7 years
T = 7
dt = T / n_steps
H = 0.1
eta = 1.5
xi_0 = 0.04
S0 = 4500

# Time grid
t = np.linspace(0, T, n_steps + 1)

# Storage
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = S0

print("⏳ Simulating rough volatility paths...")

for i in range(n_paths):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{n_paths} paths completed...")
    
    # Fractional Brownian motion
    f = FBM(n=n_steps, hurst=H, length=T, method='daviesharte')
    W_H = f.fbm()
    
    # Rough volatility path
    v_t = xi_0 * np.exp(eta * W_H - 0.5 * eta**2 * t**(2 * H))
    
    # Brownian motion for asset (vectorized)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Vectorized asset price simulation
    drift_adj = (mu - 0.5 * v_t[:-1]) * dt
    diffusion = np.sqrt(v_t[:-1]) * dW
    log_returns = drift_adj + diffusion
    
    # Use cumulative sum and exp for path generation
    paths[i, 1:] = S0 * np.exp(np.cumsum(log_returns))

print("✅ Simulation done. Saving sample plot...")

# Save 10 sample paths plot
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(t, paths[i])
plt.title(f"Sample SPX Paths under Rough Volatility (10 of 10,000)\nStart Date: {sim_date}")
plt.xlabel("Time (years)")
plt.ylabel("SPX Level")
os.makedirs("Output/simulations", exist_ok=True)
plt.savefig("Output/simulations/SPX_RoughVol_sample_plot.png")
plt.show()

# Save all paths to CSV
dates = pd.date_range(start=sim_date, periods=n_steps + 1, freq='B')
paths_df = pd.DataFrame(paths.T, index=dates)
paths_df.to_csv("Output/simulations/SPX_RoughVol_paths.csv")

print("✅ All paths saved to Output/simulations/SPX_RoughVol_paths.csv")
