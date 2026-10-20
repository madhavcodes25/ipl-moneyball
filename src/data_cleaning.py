import pandas as pd
import numpy as np

def prepare_data(file1="data/raw/IPL_Stats.csv"):
    """
    Loads and preprocesses IPL player statistics from a CSV file.
    
    This function reads the raw data, ensures the 'Year' column is numeric, 
    sorts the data to prioritize the most recent records, and removes duplicate 
    player entries to keep only their latest statistics. It also enforces 
    numeric data types for all relevant statistical and metadata columns.
    
    Args:
        file1 (str): Filepath to the raw IPL statistics CSV. Defaults to "data/raw/IPL_Stats.csv".
        
    Returns:
        pd.DataFrame: A cleaned and deduplicated DataFrame ready for analysis.
    """
    print("Loading data...")
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file1)
    
    # Convert the 'Year' column to numeric, forcing any invalid parsing (like strings) to NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Sort the DataFrame by 'Year' in descending order so the most recent stats appear at the top
    df = df.sort_values(by='Year', ascending=False)

    # Drop duplicate player entries, keeping the 'first' occurrence. 
    # Because of the descending sort above, this ensures we only keep the player's most recent year of stats.
    # .copy() is used to create a distinct DataFrame and prevent future SettingWithCopy warnings.
    df = df.drop_duplicates(subset=['Player Name'], keep='first').copy()
    
    # Define the core performance metrics that must be formatted as numbers for calculations
    numeric_cols = ['Batting Runs', 'Batting Average', 'Batting Strike Rate', 
                    'Wickets', 'Economy Rate', 'Bowling Average']
    
    # Iterate through the performance metrics, convert to numeric, and fill any missing/NaN values with 0
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Convert financial (Cost in Crores) and categorical flags (Is Foreign) to numeric types, coercing errors to NaN
    df['Cost_Cr'] = pd.to_numeric(df['Cost_Cr'], errors='coerce')
    df['Is_Foreign'] = pd.to_numeric(df['Is_Foreign'], errors='coerce')
    
    # Return the fully preprocessed DataFrame
    return df