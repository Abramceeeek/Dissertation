"""
Step 1: Data Merging and Preparation for Variable Annuity Analysis

This script merges S&P 500 option data from multiple years (2018-2023) into a single dataset
for use in the dissertation: "The Effects of Stochastic Volatility Models and Dynamic Hedging 
Strategies on Capital Requirements for Equity-Linked Variable Annuities"

Purpose: 
- Combine yearly option data files into a comprehensive dataset
- Prepare the foundation for implied volatility surface construction
- Enable analysis of market conditions across different volatility regimes

Data Sources:
- 2018: Post-financial crisis, relatively stable market conditions
- 2019: Pre-pandemic market growth period  
- 2020-2023: COVID-19 market volatility and recovery period

Author: Abdurakhmonbek Fayzullaev
"""

import pandas as pd
import os
from datetime import datetime

def merge_spx_option_data():
    """
    Merge S&P 500 option data from multiple years into a single dataset.
    
    This function combines option data spanning 2018-2023, providing a comprehensive
    view of market conditions including:
    - Normal market periods (2018-2019)
    - High volatility crisis period (2020)
    - Recovery and normalization (2021-2023)
    
    Returns:
        pandas.DataFrame: Merged option data with all years combined
    """
    
    # Define the data folder and input files
    folder = 'Data/SPX Option Chain/'
    files = [
        'SPX_Options_Data_2018.csv',
        'SPX_Options_Data_2019.csv',
        'SPX_Options_Data_2021.csv',
        'SPX_Options_Data_2022.csv',
        'SPX_Options_Data_2023.csv'
    ]
    
    print("=== S&P 500 Option Data Merging Process ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source folder: {folder}")
    
    # Load and merge all yearly data files
    merged_dataframes = []
    total_rows_per_file = []
    
    for file in files:
        file_path = os.path.join(folder, file)
        if os.path.exists(file_path):
            print(f"Loading {file}...")
            # Use low_memory=False to prevent data type warnings for large files
            df = pd.read_csv(file_path, low_memory=False)
            merged_dataframes.append(df)
            total_rows_per_file.append(len(df))
            print(f"  - Loaded {len(df):,} rows from {file}")
        else:
            print(f"WARNING: File not found - {file}")
    
    # Combine all dataframes into a single dataset
    if merged_dataframes:
        print("\nMerging all data files...")
        options_df = pd.concat(merged_dataframes, ignore_index=True)
        
        # Save the merged dataset
        output_file = os.path.join(folder, 'SPX_Options_Data_2018_to_2023_MERGED.csv')
        options_df.to_csv(output_file, index=False)
        
        # Summary statistics
        print("\n=== Merge Summary ===")
        print(f"Total files processed: {len(files)}")
        print(f"Individual file sizes: {', '.join([f'{rows:,}' for rows in total_rows_per_file])}")
        print(f"Total merged rows: {len(options_df):,}")
        print(f"Output saved to: {output_file}")
        
        # Basic data quality checks
        print("\n=== Data Quality Summary ===")
        if 'date' in options_df.columns:
            print(f"Date range: {options_df['date'].min()} to {options_df['date'].max()}")
        if 'strike' in options_df.columns:
            print(f"Strike price range: ${options_df['strike'].min():.2f} - ${options_df['strike'].max():.2f}")
        if 'implied_volatility' in options_df.columns:
            valid_iv = options_df['implied_volatility'].dropna()
            if len(valid_iv) > 0:
                print(f"IV range: {valid_iv.min():.1%} - {valid_iv.max():.1%}")
        
        print(f"\nData merging completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return options_df
    else:
        print("ERROR: No data files found for merging!")
        return None

# Execute the merging process
if __name__ == "__main__":
    merged_data = merge_spx_option_data()
