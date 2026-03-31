import pandas as pd
import numpy as np
import pulp

def calculate_fantasy(df, strategy="Balanced"):
    print("Calculating Fantasy Scores...")
    bat_weight = 1.0
    bowl_weight = 1.0
    lower_is_better = ['Economy Rate', 'Bowling Average']
    higher_is_better = ['Batting Strike Rate', 'Batting Runs', 'Batting Average', 'Wickets']
    if strategy == "Batting Heavy":
        bat_weight = 1.5
        bowl_weight = 0.5
    elif strategy == "Bowling Heavy":
        bat_weight = 0.5
        bowl_weight = 1.5

    for col in higher_is_better:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f'norm_{col}'] = (df[col] - min_val) / (max_val - min_val + 1e-6)

    for col in lower_is_better:
        valid_bowlers = df[df[col] > 0]
        if not valid_bowlers.empty:
            max_val = valid_bowlers[col].max()
            min_val = valid_bowlers[col].min()
            df[f'norm_{col}'] = np.where(df[col] > 0, (max_val - df[col]) / (max_val - min_val + 1e-6), 0)
        else:
            df[f'norm_{col}'] = 0

    df['Fantasy_Score'] = (
        df['norm_Batting Strike Rate'] * bat_weight + 
        df['norm_Batting Runs'] * bat_weight + 
        df['norm_Wickets'] * bowl_weight +
        df['norm_Economy Rate'] * bowl_weight
    ).fillna(0)
    
    return df

def optimize_team(df, budget=100.0, max_foreigners=4):
    print("Running PuLP Optimizer...")
    prob = pulp.LpProblem("IPL_Moneyball", pulp.LpMaximize)
    player_names = df['Player Name'].tolist()
    player_vars = pulp.LpVariable.dicts("Player", player_names, cat='Binary')
    fantasy_dict = dict(zip(df['Player Name'], df['Fantasy_Score']))
    cost_dict = dict(zip(df['Player Name'], df['Cost_Cr']))
    foreign_dict = dict(zip(df['Player Name'], df['Is_Foreign']))
    subrole_dict = dict(zip(df['Player Name'], df['Subrole']))
    role_dict = dict(zip(df['Player Name'], df['Role']))
    prob += pulp.lpSum([fantasy_dict[p] * player_vars[p] for p in player_names]), "Total_Fantasy"
    prob += pulp.lpSum([player_vars[p] for p in player_names]) == 11, "Must_Have_Exactly_11_Players"
    prob += pulp.lpSum([cost_dict[p] * player_vars[p] for p in player_names]) <= budget, "Stay_Under_Budget"
    prob += pulp.lpSum([foreign_dict[p] * player_vars[p] for p in player_names]) <= max_foreigners, "Max_4_Foreigners"
    prob += pulp.lpSum([player_vars[p] for p in player_names if subrole_dict.get(p) == 'WK']) >= 1, "Min_1_Wicketkeeper"
    prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict[p] in ['Bowler', 'All-Rounder']]) >= 5, "Min_5_Bowling_Options"
    prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict[p] in ['Batsman', 'All-Rounder']]) >= 6, "Min_6_Batting_Options"
    
    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("CRITICAL: Could not find an optimal solution. Your constraints might be too strict.")
        return None, 0

    selected = [p for p in player_names if player_vars[p].varValue == 1]

    final_team_df = df[df['Player Name'].isin(selected)][
        ['Player Name', 'Role', 'Subrole', 'Cost_Cr', 'Is_Foreign', 'Fantasy_Score']
    ].sort_values(by='Fantasy_Score', ascending=False)
    
    return final_team_df, prob.objective.value()
