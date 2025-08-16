# Cleans the merged dataset and adds calculated columns
import pandas as pd
import os

file_path = 'Data/SPX Option Chain/SPX_Options_Data_2018_to_2023_MERGED.csv'
output_path = 'Data/SPX Option Chain/SPX_Options_CLEANED.csv'

print("Starting data cleaning process...")
print(f"Input file: {file_path}")

# Check if input file exists
if not os.path.exists(file_path):
    print(f"ERROR: Input file not found: {file_path}")
    exit(1)

# Process data in chunks to avoid memory issues
chunk_size = 100000  # Process 100k rows at a time
chunks = []
total_rows = 0

print("Processing data in chunks...")

for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
    print(f"Processing chunk {chunk_num + 1}...")
    
    # Clean the chunk
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
    chunk['exdate'] = pd.to_datetime(chunk['exdate'], errors='coerce')
    
    # Drop rows with missing critical data
    chunk.dropna(subset=['impl_volatility', 'best_bid', 'best_offer', 'strike_price'], inplace=True)
    
    # Add calculated columns
    chunk['maturity_days'] = (chunk['exdate'] - chunk['date']).dt.days
    chunk['maturity_years'] = chunk['maturity_days'] / 365
    chunk['mid_price'] = (chunk['best_bid'] + chunk['best_offer']) / 2
    chunk['strike'] = chunk['strike_price'] / 1000
    
    chunks.append(chunk)
    total_rows += len(chunk)
    
    print(f"  Chunk {chunk_num + 1}: {len(chunk):,} rows processed")

# Combine all chunks
print("Combining processed chunks...")
cleaned_df = pd.concat(chunks, ignore_index=True)

# Save cleaned data
print(f"Saving cleaned data to: {output_path}")
cleaned_df.to_csv(output_path, index=False)

print(f"\nData cleaning completed successfully!")
print(f"Total rows processed: {total_rows:,}")
print(f"Output saved to: {output_path}")
print(f"Final dataset shape: {cleaned_df.shape}")
