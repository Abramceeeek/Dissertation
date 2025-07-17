import pandas as pd

# Load full risk-free data
df = pd.read_csv("Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv")

# Rename for clarity (optional but helpful)
df = df.rename(columns={"days": "maturity_days"})

# Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Keep only needed columns
clean_df = df[['date', 'maturity_days', 'rate']]

# Save cleaned file
clean_df.to_csv("Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv", index=False)

print("✅ Cleaned and saved risk-free curve data to '...CLEANED.csv'")
