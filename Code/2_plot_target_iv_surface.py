import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the snapshot data you saved
snapshot_date = '2018-06-01'
df_snapshot = pd.read_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv')

print(f"✅ Loaded snapshot for {snapshot_date} with {len(df_snapshot)} rows")

# Quick check
print(df_snapshot.head())

# Plotting
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Create grid
ax.scatter(
    df_snapshot['strike'],
    df_snapshot['maturity_days'],
    df_snapshot['impl_volatility'],
    c=df_snapshot['impl_volatility'],
    cmap=cm.viridis,
    alpha=0.8
)

ax.set_title(f"Market Implied Volatility Surface – {snapshot_date}")
ax.set_xlabel('Strike Price')
ax.set_ylabel('Maturity (days)')
ax.set_zlabel('Implied Volatility')
plt.show()
