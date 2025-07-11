import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import os

# Load cleaned option data
df = pd.read_csv('Data/SPX Option Chain/SPX_Options_CLEANED.csv', low_memory=False)

print("Unique dates in dataset:", df['date'].unique())

dates = [
    '2018-01-03',
    '2018-06-01',
    '2018-12-01',
    '2019-01-03',
    '2019-06-01',
    '2019-12-01',
    '2020-01-03',
    '2020-06-01',
    '2020-12-01',
    '2021-01-03',
    '2021-06-01',
    '2021-12-01',
    '2022-01-03',
    '2022-06-01',
    '2022-12-01',
    '2023-01-03',
    '2023-06-01',
    '2023-12-01'
]

for snapshot_date in dates:
    print(f"\n✅ Processing snapshot date: {snapshot_date}")

    subset_df = df[df['date'] == snapshot_date].copy()

    if subset_df.empty:
        print(f"⚠️ No data for {snapshot_date}, skipping...")
        continue

    subset_df = subset_df[
        (subset_df['cp_flag'].str.lower() == 'c') &
        (subset_df['impl_volatility'].astype(float) > 0.01) &
        (subset_df['impl_volatility'].astype(float) < 2.0) &
        (subset_df['strike'].astype(float) > 100) &
        (subset_df['maturity_days'].astype(float) <= 365)
    ]

    if subset_df.empty:
        print(f"⚠️ No valid call options after filtering for {snapshot_date}, skipping...")
        continue

    print(f"✅ Filtered rows for {snapshot_date}: {len(subset_df)}")

    # Create meshgrid
    strikes = np.linspace(subset_df['strike'].min(), subset_df['strike'].max(), 100)
    maturities = np.linspace(subset_df['maturity_days'].min(), subset_df['maturity_days'].max(), 100)
    X, Y = np.meshgrid(strikes, maturities)

    Z = griddata(
        (subset_df['strike'], subset_df['maturity_days']),
        subset_df['impl_volatility'],
        (X, Y),
        method='linear'
    )

    mask = ~np.isnan(Z)
    if not np.any(mask):
        print(f"⚠️ Could not interpolate surface for {snapshot_date}, skipping...")
        continue

    X_masked = X[mask]
    Y_masked = Y[mask]
    Z_masked = Z[mask]

    Z_filled = griddata(
        (X_masked, Y_masked),
        Z_masked,
        (X, Y),
        method='nearest'
    )

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_filled, cmap=cm.viridis, edgecolor='none')
    ax.set_title(f'Implied Volatility Surface – {snapshot_date}')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Maturity (days)')
    ax.set_zlabel('Implied Volatility')

    # Ensure folder exists
    os.makedirs('Output/surfaces', exist_ok=True)
    plot_path = f'Output/surfaces/iv_surface_{snapshot_date}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Vol surface saved to: {plot_path}")
