
import pandas as pd

# Load the cleaned options data
cleaned_file_path = 'Data/SPX Option Chain/SPX_Options_CLEANED.csv'
options_df = pd.read_csv(cleaned_file_path, low_memory=False)

# Convert 'date' column to datetime objects
options_df['date'] = pd.to_datetime(options_df['date'])

# Define the snapshot date
snapshot_date = '2018-06-01'

# Filter the dataframe for the snapshot date
df_snapshot = options_df[options_df['date'] == snapshot_date].copy()

# Save the snapshot to a new CSV file
snapshot_output_path = f'Data/SPX_Snapshot_{snapshot_date}.csv'
df_snapshot.to_csv(snapshot_output_path, index=False)

print(f"Successfully created snapshot for {snapshot_date} with {len(df_snapshot)} rows.")
print(f"Snapshot saved to: {snapshot_output_path}")
