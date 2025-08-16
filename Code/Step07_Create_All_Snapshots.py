
import pandas as pd

# Load the cleaned options data
cleaned_file_path = 'Data/SPX Option Chain/SPX_Options_CLEANED.csv'
options_df = pd.read_csv(cleaned_file_path, low_memory=False)

# Convert 'date' column to datetime objects
options_df['date'] = pd.to_datetime(options_df['date'])

# List of snapshot dates
snapshot_dates = [
    '2020-06-01',
    '2020-12-01',
    '2021-06-01',
    '2022-01-03',
    '2022-06-01',
    '2023-01-03',
    '2023-06-01'
]

# Loop through the dates and create snapshots
for snapshot_date in snapshot_dates:
    df_snapshot = options_df[options_df['date'] == snapshot_date].copy()
    if not df_snapshot.empty:
        snapshot_output_path = f'Data/SPX_Snapshot_{snapshot_date}.csv'
        df_snapshot.to_csv(snapshot_output_path, index=False)
        print(f"Successfully created snapshot for {snapshot_date} with {len(df_snapshot)} rows.")
        print(f"Snapshot saved to: {snapshot_output_path}")
    else:
        print(f"No data found for {snapshot_date}. Snapshot not created.")
