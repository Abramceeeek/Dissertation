import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rila.models import simulate_gbm
from rila.config import S0, n_paths, T, N

seed = 42
mu = 0.01  
sigma = 0.2 
S = simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=seed)

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
