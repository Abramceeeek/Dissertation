# Cleans the merged dataset and adds calculated columns
import pandas as pd

file_path = 'Data/SPX Option Chain/SPX_Options_Data_2018_to_2023_MERGED.csv'
options_df = pd.read_csv(file_path, low_memory=False)

# Preprocess
options_df['date'] = pd.to_datetime(options_df['date'], errors='coerce')
options_df['exdate'] = pd.to_datetime(options_df['exdate'], errors='coerce')

options_df.dropna(subset=['impl_volatility', 'best_bid', 'best_offer', 'strike_price'], inplace=True)

options_df['maturity_days'] = (options_df['exdate'] - options_df['date']).dt.days
options_df['maturity_years'] = options_df['maturity_days'] / 365
options_df['mid_price'] = (options_df['best_bid'] + options_df['best_offer']) / 2
options_df['strike'] = options_df['strike_price'] / 1000

# Save clean version
options_df.to_csv('Data/SPX Option Chain/SPX_Options_CLEANED.csv', index=False)

print("✅ Data cleaned and saved:")
