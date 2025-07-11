# Merges individual yearly files into one
import pandas as pd
import os

folder = 'Data/SPX Option Chain/'
files = ['SPX_Options_Data_2018.csv', 'SPX_Options_Data_2019.csv', 'SPX_Options_Data_2020_to_2023.csv']
dfs = [pd.read_csv(os.path.join(folder, f), low_memory=False) for f in files]

options_df = pd.concat(dfs, ignore_index=True)
options_df.to_csv(os.path.join(folder, 'SPX_Options_Data_2018_to_2023_MERGED.csv'), index=False)

print("✅ Data merged successfully. Rows:", len(options_df))
