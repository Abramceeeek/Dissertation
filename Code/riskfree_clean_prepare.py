import pandas as pd

df = pd.read_csv("Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023.csv")

df = df.rename(columns={"days": "maturity_days"})

df['date'] = pd.to_datetime(df['date'], errors='coerce')

clean_df = df[['date', 'maturity_days', 'rate']]

clean_df.to_csv("Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv", index=False)

print("Cleaned and saved risk-free curve data to '...CLEANED.csv'")
