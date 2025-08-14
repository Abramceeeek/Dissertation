import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rila.models import simulate_heston
from rila.config import S0, n_paths, T, N, heston_params

seed = 42
v0 = heston_params['v0']
mu = 0.01
kappa = heston_params['kappa']
theta = heston_params['theta']
sigma_v = heston_params['sigma_v']
rho = heston_params['rho']

S = simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, T, N, n_paths, seed=seed)

import datetime
start_date = datetime.date(2023, 1, 1)
dates = pd.bdate_range(start=start_date, periods=N + 1)
sample_paths = pd.DataFrame(S[:, :10], index=dates)

os.makedirs("Output/simulations", exist_ok=True)
pd.DataFrame(S, index=dates).to_csv("Output/simulations/SPX_Heston_paths.csv")

plt.figure(figsize=(12, 6))
plt.plot(sample_paths)
plt.title("Sample SPX Paths under Heston Model (10 of 10,000)")
plt.xlabel("Date")
plt.ylabel("SPX Level")
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/simulations/SPX_Heston_sample_plot.png")
plt.show()

print("Heston simulation complete. Paths and plot saved.")
