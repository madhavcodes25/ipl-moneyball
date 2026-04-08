import pandas as pd
import numpy as np

def prepare_data(file1="data/raw/IPL_Stats.csv"):
    print("Loading data...")
    # Load raw data from CSV
    df = pd.read_csv(file1)

    # Convert 'Year' to numeric, replacing errors with NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Sort data by year in descending order (newest stats first)
    df = df.sort_values(by='Year', ascending=False)

    # Keep only the most recent entry for each player
    df = df.drop_duplicates(subset=['Player Name'], keep='first').copy()

    # Define performance columns requiring numeric conversion
    numeric_cols = ['Batting Runs', 'Batting Average', 'Batting Strike Rate', 
                    'Wickets', 'Economy Rate', 'Bowling Average']

    # Convert performance columns to numeric and replace NaNs with 0
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert financial and demographic columns to numeric
    df['Cost_Cr'] = pd.to_numeric(df['Cost_Cr'], errors='coerce')
    df['Is_Foreign'] = pd.to_numeric(df['Is_Foreign'], errors='coerce')
    
    return df
