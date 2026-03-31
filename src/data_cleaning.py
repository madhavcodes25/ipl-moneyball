import pandas as pd
import numpy as np

def prepare_data(file1="data/raw/IPL_Stats.csv"):
    print("Loading data...")
    df = pd.read_csv(file1)
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    df = df.sort_values(by='Year', ascending=False)

    df = df.drop_duplicates(subset=['Player Name'], keep='first').copy()
    
    numeric_cols = ['Batting Runs', 'Batting Average', 'Batting Strike Rate', 
                    'Wickets', 'Economy Rate', 'Bowling Average']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['Cost_Cr'] = pd.to_numeric(df['Cost_Cr'], errors='coerce')
    df['Is_Foreign'] = pd.to_numeric(df['Is_Foreign'], errors='coerce')
    
    return df
