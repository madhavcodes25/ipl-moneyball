import pandas as pd
import numpy as np
import pulp

def calculate_fantasy(df, strategy="Balanced", min_runs=100, min_wickets=5):
    print("Calculating Fantasy Ratings...")

    # Set multipliers based on the chosen strategy
    bat_weight = 1.2 if strategy == "Batting Heavy" else (0.8 if strategy == "Bowling Heavy" else 1.0)
    bowl_weight = 1.2 if strategy == "Bowling Heavy" else (0.8 if strategy == "Batting Heavy" else 1.0)

    batting_metrics = ['Batting Runs', 'Batting Strike Rate', 'Batting Average']
    bowling_metrics = ['Wickets', 'Economy Rate', 'Bowling Average']

    # Filter players who meet the minimum criteria to avoid skewing stats
    qualified_bat = df[df['Batting Runs'] >= min_runs]
    qualified_bowl = df[df['Wickets'] >= min_wickets]

    # Normalize batting metrics (higher is better)
    for col in batting_metrics:
        if not qualified_bat.empty:
            max_val = qualified_bat[col].max()
            min_val = qualified_bat[col].min()

            # Scale between 0 and 1
            df[f'norm_{col}'] = ((df[col] - min_val) / (max_val - min_val + 1e-6)).clip(0, 1)

            # Penalize strike rate and average if total runs are very low
            if col in ['Batting Strike Rate', 'Batting Average']:
                df[f'norm_{col}'] = np.where(df['Batting Runs'] < 30, df[f'norm_{col}'] * 0.2, df[f'norm_{col}'])
        else:
            df[f'norm_{col}'] = 0

    # Normalize Wickets (higher is better)
    if not qualified_bowl.empty:
        w_max, w_min = qualified_bowl['Wickets'].max(), qualified_bowl['Wickets'].min()
        df['norm_Wickets'] = ((df['Wickets'] - w_min) / (w_max - w_min + 1e-6)).clip(0, 1)
    else:
        df['norm_Wickets'] = 0

    # Normalize Economy Rate and Bowling Average (lower is better)
    for col in ['Economy Rate', 'Bowling Average']:
        if not qualified_bowl.empty:
            max_val = qualified_bowl[col].max()
            min_val = qualified_bowl[col].min()

            # Invert scale so lower values get higher normalized scores
            df[f'norm_{col}'] = ((max_val - df[col]) / (max_val - min_val + 1e-6)).clip(0, 1)

            # Penalize economy and average if wickets taken are very low
            df[f'norm_{col}'] = np.where(df['Wickets'] < 2, df[f'norm_{col}'] * 0.2, df[f'norm_{col}'])
        else:
            df[f'norm_{col}'] = 0

    # Calculate final weighted fantasy score
    df['Fantasy_Score'] = (
        (df['norm_Batting Runs'] * 1.5 + df['norm_Batting Strike Rate'] * 0.75 + df['norm_Batting Average'] * 0.75) * bat_weight + 
        (df['norm_Wickets'] * 1.5 + df['norm_Economy Rate'] * 0.75 + df['norm_Bowling Average'] * 0.75) * bowl_weight
    ).fillna(0)
    
    return df

def is_pacer(subrole_str):
    """Safely checks if a player is a fast/medium bowler based on their subrole text."""
    s = str(subrole_str).lower()
    return any(keyword in s for keyword in ['pace', 'fast', 'medium', 'seam'])

def is_spinner(subrole_str):
    """Safely checks if a player is a spin bowler based on their subrole text."""
    s = str(subrole_str).lower()
    return any(keyword in s for keyword in ['spin', 'orthodox', 'leg', 'googly', 'carrom'])

def optimize_team(df, budget=100.0, max_foreigners=4, must_include=None,must_exclude=None,retention_prices=None,custom_auction_prices=None,
                  constraint_mode="Flexible (Auto-balance)", 
                  num_batters=4, num_bowlers=4, num_all_rounders=2, 
                  num_pacers=3, num_spinners=2):
    print("Running PuLP Optimizer...")


    # Initialize default empty lists/dicts if None                  
    if must_include is None:
        must_include = []
    if must_exclude is None: 
        must_exclude = []    
    if retention_prices is None: 
        retention_prices = {}    
    if custom_auction_prices is None: 
        custom_auction_prices = {}    


    # Initialize maximization problem                   
    prob = pulp.LpProblem("IPL_Moneyball", pulp.LpMaximize)

    # Create binary variables for each player (1 if selected, 0 if not)                  
    player_names = df['Player Name'].tolist()
    player_vars = pulp.LpVariable.dicts("Player", player_names, cat='Binary')

    # Create lookup dictionaries for quick access                  
    fantasy_dict = dict(zip(df['Player Name'], df['Fantasy_Score']))
    cost_dict = dict(zip(df['Player Name'], df['Cost_Cr']))

    # Apply custom prices for retained or manually priced players                  
    for player, custom_price in retention_prices.items():
        if player in cost_dict:
            cost_dict[player] = custom_price
    for player, custom_price in custom_auction_prices.items():
        if player in cost_dict:
            cost_dict[player] = custom_price        

    foreign_dict = dict(zip(df['Player Name'], df['Is_Foreign']))
    subrole_dict = dict(zip(df['Player Name'], df['Subrole']))
    role_dict = dict(zip(df['Player Name'], df['Role']))
    # Objective Function: Maximize total fantasy score of selected players
    prob += pulp.lpSum([fantasy_dict[p] * player_vars[p] for p in player_names]), "Total_Fantasy"
    # Core Constraints
    prob += pulp.lpSum([player_vars[p] for p in player_names]) == 11, "Must_Have_Exactly_11_Players"
    prob += pulp.lpSum([cost_dict[p] * player_vars[p] for p in player_names]) <= budget, "Stay_Under_Budget"
    prob += pulp.lpSum([foreign_dict[p] * player_vars[p] for p in player_names]) <= max_foreigners, "Max_4_Foreigners"
    # Apply role-based constraints depending on the selected mode
    if constraint_mode == "Strict (Custom Roles)":
        
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'Batsman']) == num_batters, "Exact_Batsmen"
        prob += pulp.lpSum([player_vars[p] for p in player_names if subrole_dict.get(p) == 'WK']) >= 1, "Min_1_WicketKeeper"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'Bowler']) == num_bowlers, "Exact_Bowlers"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'All-Rounder']) == num_all_rounders, "Exact_AllRounders"

        
        prob += pulp.lpSum([player_vars[p] for p in player_names if is_pacer(subrole_dict.get(p))]) >= num_pacers, "Min_Pacers"
        prob += pulp.lpSum([player_vars[p] for p in player_names if is_spinner(subrole_dict.get(p))]) >= num_spinners, "Min_Spinners"

    else:
        # Flexible constraints
        prob += pulp.lpSum([player_vars[p] for p in player_names if subrole_dict.get(p) == 'WK']) >= 1, "Min_1_Wicketkeeper"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) in ['Bowler', 'All-Rounder']]) >= 5, "Min_5_Bowling_Options"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) in ['Batsman', 'All-Rounder']]) >= 6, "Min_6_Batting_Options"
    # Apply manual inclusions (force selection)
    for player in must_include:
        if player in player_vars:
            constraint_name = f"Force_Include_{player.replace(' ', '_').replace('-', '_')}"
            prob += player_vars[player] == 1, constraint_name
    # Apply manual exclusions (prevent selection)
    for player in must_exclude:
        if player in player_vars:
            constraint_name = f"Force_Exclude_{player.replace(' ', '_').replace('-', '_')}"
            prob += player_vars[player] == 0, constraint_name        
    # Solve the optimization problem        
    prob.solve()
    # Handle failure to find a valid team
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("CRITICAL: Could not find an optimal solution. Your constraints might be too strict.")
        return None, 0
    # Extract selected players and generate final dataframe
    selected = [p for p in player_names if player_vars[p].varValue == 1]

    final_team_df = df[df['Player Name'].isin(selected)][
        ['Player Name', 'Role', 'Subrole', 'Cost_Cr', 'Is_Foreign', 'Fantasy_Score']
    ].sort_values(by='Fantasy_Score', ascending=False)
    
    return final_team_df, prob.objective.value()
