import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import os

# Load cleaned dataset
df_all = pd.read_csv('Data/SPX Option Chain/SPX_Options_CLEANED.csv', low_memory=False)

# Make sure output folders exist
os.makedirs('Output/surfaces', exist_ok=True)
os.makedirs('Output/logs', exist_ok=True)

# Get all unique dates
dates = df_all['date'].unique()
print(f"🔄 Processing {len(dates)} unique snapshot dates...\n")

# Loop over each snapshot date
for snapshot_date in dates:
    df = df_all[df_all['date'] == snapshot_date].copy()

    # Apply safe filtering
    df = df[
        (df['cp_flag'].str.lower() == 'c') &
        (df['impl_volatility'].astype(float) > 0.01) &
        (df['impl_volatility'].astype(float) < 2.0) &
        (df['strike'].astype(float) > 100) &  # relaxed
        (df['maturity_days'].astype(float) <= 365)
    ]

    if df.empty or len(df) < 20:
        print(f"⛔ {snapshot_date}: Not enough usable data ({len(df)} rows). Skipping.")
        with open('Output/logs/skipped_dates.txt', 'a') as f:
            f.write(f"{snapshot_date}\n")
        continue

    # Create meshgrid for interpolation
    strikes = np.linspace(df['strike'].min(), df['strike'].max(), 100)
    maturities = np.linspace(df['maturity_days'].min(), df['maturity_days'].max(), 100)
    X, Y = np.meshgrid(strikes, maturities)

    try:
        # Interpolate using griddata
        Z = griddata(
            (df['strike'], df['maturity_days']),
            df['impl_volatility'],
            (X, Y),
            method='linear'
        )

        # Re-interpolate to fill NaNs (using nearest)
        mask = ~np.isnan(Z)
        Z_filled = griddata(
            (X[mask], Y[mask]),
            Z[mask],
            (X, Y),
            method='nearest'
        )

        # Plot the surface
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_filled, cmap=cm.viridis, edgecolor='none')
        ax.set_title(f'Implied Volatility Surface – {snapshot_date}')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Maturity (days)')
        ax.set_zlabel('Implied Volatility')

        # Save the plot
        save_path = f'Output/surfaces/iv_surface_{snapshot_date}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✅ {snapshot_date}: Surface saved to {save_path}")

    except Exception as e:
        print(f"⚠️ {snapshot_date}: Failed to plot due to {e}")
        with open('Output/logs/errors.txt', 'a') as f:
            f.write(f"{snapshot_date} – {str(e)}\n")
