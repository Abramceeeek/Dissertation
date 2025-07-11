import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
df = pd.read_csv('Data/SPX Option Chain/SPX_Options_CLEANED.csv')

# Dates to analyze (pick 5–10 representative ones)
dates = [
    '2020-06-01',
    '2020-12-01',
    '2021-06-01',
    '2022-01-03',
    '2022-06-01',
    '2023-01-03',
    '2023-06-01'
]

# Ensure output folder exists
os.makedirs('Output/plots', exist_ok=True)

for date in dates:
    subset = df[
        (df['date'] == date) &
        (df['maturity_days'].between(25, 35)) &
        (df['cp_flag'] == 'C')
    ]

    if subset.empty:
        print(f"⚠️ No data for {date}, skipping...")
        continue

    plt.figure(figsize=(10, 5))
    plt.scatter(subset['strike'], subset['impl_volatility'], alpha=0.6, color='blue')
    plt.title(f'IV Smile – SPX Calls ~30D Maturity on {date}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid(True)

    save_path = f'Output/plots/iv_smile_{date}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")
