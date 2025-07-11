import pandas as pd

# Load full risk-free curve data
df = pd.read_csv('Data/Risk-Free Yield Curve/Interest_Rate_Curves_2000_2023.csv')

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# Filter to 2018-01-01 through 2023-12-31
mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2023-12-31')
df_filtered = df.loc[mask]

print(f"✅ Filtered rows: {len(df_filtered)}")

# Save trimmed version
df_filtered.to_csv('Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv', index=False)
print("✅ Saved to Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv")
