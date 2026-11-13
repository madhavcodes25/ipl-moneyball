import pandas as pd
import numpy as np
import pulp
from typing import Optional

def calculate_fantasy(df, strategy="Balanced", min_runs=100, min_wickets=5):
    """
    Calculates a composite Fantasy Score for each player based on their career statistics
    (Batting: Runs, Strike Rate, Average; Bowling: Wickets, Economy Rate, Bowling Average).

    Phase Specialists & Phase Score
    -------------------------------
    Phase-specific composite scores (Powerplay, Middle, Death) and an aggregate ``Phase_Score``
    are computed for phase ranking in the Phase Analysis tab and for assigning specialist role tags:
    ``pp_bat_specialist``, ``death_bat_specialist``, ``pp_bowl_specialist``, ``death_bowl_specialist``.
    Phase score is NOT blended into Fantasy_Score, keeping overall player valuation grounded in
    robust career output while allowing the LP optimizer to enforce specialist constraints when selected.

    Args:
        df (pd.DataFrame): DataFrame containing player statistics.
        strategy (str): Weighting strategy for the score ("Balanced", "Batting Heavy", or "Bowling Heavy").
        min_runs (int): Minimum runs required to be considered a 'qualified' batter for normalization boundaries.
        min_wickets (int): Minimum wickets required to be considered a 'qualified' bowler for normalization boundaries.

    Returns:
        pd.DataFrame: The original DataFrame with normalized metric columns, the final 'Fantasy_Score',
                      'Value_Score', 'Phase_Score', and specialist boolean columns.
    """
    print("Calculating Fantasy Ratings...")

    # Determine the multipliers for batting and bowling metrics based on the chosen strategy
    bat_weight = 1.2 if strategy == "Batting Heavy" else (0.8 if strategy == "Bowling Heavy" else 1.0)
    bowl_weight = 1.2 if strategy == "Bowling Heavy" else (0.8 if strategy == "Batting Heavy" else 1.0)

    # Define the specific columns to be evaluated for batters and bowlers
    batting_metrics = ['Batting Runs', 'Batting Strike Rate', 'Batting Average']
    bowling_metrics = ['Wickets', 'Economy Rate', 'Bowling Average']

    # Filter players who meet the minimum threshold to establish robust min/max normalization bounds
    qualified_bat = df[df['Batting Runs'] >= min_runs]
    qualified_bowl = df[df['Wickets'] >= min_wickets]

    # Normalize Batting Metrics
    for col in batting_metrics:
        if not qualified_bat.empty:
            max_val = qualified_bat[col].max()
            min_val = qualified_bat[col].min()
            df[f'norm_{col}'] = ((df[col] - min_val) / (max_val - min_val + 1e-6)).clip(0, 1)
            # Penalize tail-enders with inflated stats from very few runs
            if col in ['Batting Strike Rate', 'Batting Average']:
                df[f'norm_{col}'] = np.where(df['Batting Runs'] < 30, df[f'norm_{col}'] * 0.2, df[f'norm_{col}'])
        else:
            df[f'norm_{col}'] = 0

    # Normalize Bowling Metric: Wickets
    if not qualified_bowl.empty:
        w_max, w_min = qualified_bowl['Wickets'].max(), qualified_bowl['Wickets'].min()
        df['norm_Wickets'] = ((df['Wickets'] - w_min) / (w_max - w_min + 1e-6)).clip(0, 1)
    else:
        df['norm_Wickets'] = 0

    # Normalize Bowling Metrics: Economy Rate & Bowling Average (lower = better → invert)
    for col in ['Economy Rate', 'Bowling Average']:
        if not qualified_bowl.empty:
            max_val = qualified_bowl[col].max()
            min_val = qualified_bowl[col].min()
            df[f'norm_{col}'] = ((max_val - df[col]) / (max_val - min_val + 1e-6)).clip(0, 1)
            df[f'norm_{col}'] = np.where(df['Wickets'] < 2, df[f'norm_{col}'] * 0.2, df[f'norm_{col}'])
        else:
            df[f'norm_{col}'] = 0

    # --- BASE FANTASY SCORE ---
    base_score = (
        (df['norm_Batting Runs'] * 1.5 + df['norm_Batting Strike Rate'] * 0.75 + df['norm_Batting Average'] * 0.75) * bat_weight +
        (df['norm_Wickets'] * 1.5 + df['norm_Economy Rate'] * 0.75 + df['norm_Bowling Average'] * 0.75) * bowl_weight
    ).fillna(0)

    # --- PHASE SPECIALIST BONUS ---
    # Composite specialist scores per phase:
    #
    #   Batting  = 0.4 × norm(phase_SR) + 0.3 × norm(phase_runs) + 0.3 × norm(phase_avg)
    #     - All stats are phase-specific (PP / Middle / Death), not career averages.
    #     - SR:   scoring rate / aggression in that phase
    #     - Runs: volume / consistency in that phase
    #     - Avg:  quality — penalises cheap dismissals in that phase
    #
    #   Bowling  = 0.5 × norm(phase_wkts) + 0.5 × norm(inv_phase_econ)
    #     - Both stats are phase-specific.
    #     - phase_wkts: total wickets taken IN that phase — primary signal (Bhuvi 44 >> Bracewell 3)
    #     - inv_econ:   containment in that phase
    #     - wkts_per_over was removed: it is derived from the same data as wkts+balls and
    #       disproportionately inflates small-sample players.

    def _minmax_norm(series: pd.Series, invert: bool = False) -> pd.Series:
        """Min-max normalize a series to [0,1]; invert for 'lower is better' metrics."""
        valid = series.dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=series.index)
        mn, mx = valid.min(), valid.max()
        if invert:
            normed = ((mx - series) / (mx - mn + 1e-6)).clip(0, 1)
        else:
            normed = ((series - mn) / (mx - mn + 1e-6)).clip(0, 1)
        return normed.fillna(0)

    # --- Batting composite per phase: SR + Runs + Average (all phase-specific) ---
    bat_phase_map = {
        "pp":     ("pp_bat_sr",     "pp_bat_runs",     "pp_bat_avg"),
        "middle": ("middle_bat_sr", "middle_bat_runs", "middle_bat_avg"),
        "death":  ("death_bat_sr",  "death_bat_runs",  "death_bat_avg"),
    }
    bat_composite = {}
    for phase_key, (sr_col, runs_col, avg_col) in bat_phase_map.items():
        sr_norm   = _minmax_norm(df[sr_col])   if sr_col   in df.columns else pd.Series(0.0, index=df.index)
        runs_norm = _minmax_norm(df[runs_col]) if runs_col in df.columns else pd.Series(0.0, index=df.index)
        avg_norm  = _minmax_norm(df[avg_col])  if avg_col  in df.columns else pd.Series(0.0, index=df.index)

        if runs_col in df.columns and avg_col in df.columns:
            bat_composite[phase_key] = 0.4 * sr_norm + 0.3 * runs_norm + 0.3 * avg_norm
        elif runs_col in df.columns:
            bat_composite[phase_key] = 0.6 * sr_norm + 0.4 * runs_norm
        else:
            bat_composite[phase_key] = sr_norm  # fallback

    # --- Bowling composite per phase: phase wickets (50%) + inv phase economy (50%) ---
    # Both metrics are phase-specific — economy in THAT phase, wickets in THAT phase.
    # wkts_per_over removed: redundant with wkts+balls and inflates small-sample players.
    bowl_phase_map = {
        "pp":     ("pp_bowl_econ",     "pp_bowl_wkts"),
        "middle": ("middle_bowl_econ", "middle_bowl_wkts"),
        "death":  ("death_bowl_econ",  "death_bowl_wkts"),
    }
    bowl_composite = {}
    for phase_key, (econ_col, wkts_col) in bowl_phase_map.items():
        econ_norm = _minmax_norm(df[econ_col], invert=True) if econ_col in df.columns else pd.Series(0.0, index=df.index)
        wkts_norm = _minmax_norm(df[wkts_col])              if wkts_col in df.columns else pd.Series(0.0, index=df.index)

        if wkts_col in df.columns:
            # Equal weight: 50% phase wickets + 50% phase economy
            bowl_composite[phase_key] = 0.5 * wkts_norm + 0.5 * econ_norm
        else:
            bowl_composite[phase_key] = econ_norm

    # Phase_Score = average of all 6 composite phases (3 bat + 3 bowl)
    all_composites = list(bat_composite.values()) + list(bowl_composite.values())
    phase_matrix = pd.concat(all_composites, axis=1)
    df["Phase_Score"] = phase_matrix.mean(axis=1).fillna(0).round(3)

    # Fantasy Score is purely stats-based (no phase score blended)
    # Phase_Score is strictly used for phase ranking and specialist identification
    df["Fantasy_Score"] = base_score.fillna(0)

    # --- SPECIALIST ROLE TAGS (top-quartile composite per phase) ---
    specialist_map = [
        ("pp_bat_specialist",    bat_composite["pp"]),
        ("death_bat_specialist", bat_composite["death"]),
        ("pp_bowl_specialist",   bowl_composite["pp"]),
        ("death_bowl_specialist",bowl_composite["death"]),
    ]
    for tag, composite_series in specialist_map:
        has_data = composite_series > 0
        if has_data.any():
            threshold = composite_series[has_data].quantile(0.75)
            df[tag] = has_data & (composite_series >= threshold)
        else:
            df[tag] = False

    # --- VALUE SCORE (The Moneyball Metric) ---
    # Value Score = Fantasy performance per crore spent.
    df['Value_Score'] = np.where(
        df['Cost_Cr'] > 0,
        df['Fantasy_Score'] / df['Cost_Cr'],
        0
    )

    return df


def _get_solver():
    """
    Returns the best available PuLP solver on this machine.

    Priority order:
    1. System CBC (e.g., installed via `brew install cbc`) — fast and native.
    2. Bundled PuLP CBC binary — may be unavailable on some architectures (e.g., ARM macOS).
    3. GLPK (if installed separately).
    Raises a RuntimeError with a helpful message if no solver is found.
    """
    import shutil

    # Try system CBC first (native, works on all architectures)
    system_cbc = shutil.which("cbc")
    if system_cbc:
        return pulp.COIN_CMD(path=system_cbc, msg=0)

    # Try the PuLP-bundled CBC (works on x86; fails on ARM macOS with old PuLP)
    try:
        solver = pulp.PULP_CBC_CMD(msg=0)
        # Quick sanity check: attempt to solve a trivial problem to verify binary works
        _test = pulp.LpProblem("solver_check", pulp.LpMaximize)
        _x = pulp.LpVariable("x", 0, 1)
        _test += _x
        _test.solve(solver)
        return solver
    except Exception:
        pass

    # Try GLPK as last resort
    try:
        glpk = pulp.GLPK_CMD(msg=0)
        return glpk
    except Exception:
        pass

    raise RuntimeError(
        "No working LP solver found. Install CBC via Homebrew (`brew install cbc`) "
        "or install GLPK (`brew install glpk`)."
    )


def is_pacer(subrole_str):
    """Safely checks if a player is a fast/medium bowler based on their subrole text."""
    s = str(subrole_str).lower()
    return any(keyword in s for keyword in ['pace', 'fast', 'medium', 'seam'])


def is_spinner(subrole_str):
    """Safely checks if a player is a spin bowler based on their subrole text."""
    s = str(subrole_str).lower()
    return any(keyword in s for keyword in ['spin', 'orthodox', 'leg', 'googly', 'carrom'])


def optimize_team(df, budget=100.0, max_foreigners=4, must_include=None, must_exclude=None,
                  retention_prices=None, custom_auction_prices=None,
                  constraint_mode="Flexible (Auto-balance)",
                  num_batters=4, num_bowlers=4, num_all_rounders=2,
                  num_pacers=3, num_spinners=2,
                  require_pp_bat_specialist=False,
                  require_death_bat_specialist=False,
                  require_pp_bowl_specialist=False,
                  require_death_bowl_specialist=False):
    """
    Uses Linear Programming (PuLP) to select the optimal 11-player team that maximizes Fantasy Score
    while strictly adhering to budget, foreigner limits, team composition rules, and optional
    phase specialist requirements.

    Args:
        df (pd.DataFrame): DataFrame containing players, their costs, roles, and calculated Fantasy Scores.
        budget (float): Maximum allowed budget to spend on the 11 players.
        max_foreigners (int): Maximum number of overseas players allowed in the XI.
        must_include (list): List of player names strictly required in the final team.
        must_exclude (list): List of player names to be completely ignored by the optimizer.
        retention_prices (dict): Mapping of {Player Name: Retained Price} to override default dataset costs.
        custom_auction_prices (dict): Mapping of {Player Name: Expected Auction Price} to override default costs.
        constraint_mode (str): "Strict (Custom Roles)" for exact role counts, or "Flexible (Auto-balance)" for broad minimums.
        num_batters (int): Exact number of batsmen (used in Strict mode).
        num_bowlers (int): Exact number of bowlers (used in Strict mode).
        num_all_rounders (int): Exact number of all-rounders (used in Strict mode).
        num_pacers (int): Minimum number of pace bowlers (used in Strict mode).
        num_spinners (int): Minimum number of spin bowlers (used in Strict mode).
        require_pp_bat_specialist (bool): If True, force at least one Powerplay batting specialist in the XI.
        require_death_bat_specialist (bool): If True, force at least one Death Overs batting specialist in the XI.
        require_pp_bowl_specialist (bool): If True, force at least one Powerplay bowling specialist in the XI.
        require_death_bowl_specialist (bool): If True, force at least one Death Overs bowling specialist in the XI.

    Returns:
        tuple: (Filtered pd.DataFrame of the selected 11 players, Maximum objective/fantasy value achieved)
    """
    print("Running PuLP Optimizer...")
    
    # Initialize mutable default arguments as empty data structures to prevent state leakage
    if must_include is None:
        must_include = []
    if must_exclude is None: 
        must_exclude = []    
    if retention_prices is None: 
        retention_prices = {}    
    if custom_auction_prices is None: 
        custom_auction_prices = {}    

    # Define the Linear Programming problem seeking to Maximize the objective function (Fantasy Score)
    prob = pulp.LpProblem("IPL_Moneyball", pulp.LpMaximize)
    
    # Create a binary decision variable for every player (1 if selected, 0 if not selected)
    # Using explicit dict comprehension is robust across all PuLP versions (v2, v3, v4)
    # and avoids PuLP 4.x AttributeError with LpVariable.dicts deprecation.
    player_names = df['Player Name'].tolist()
    player_vars = {
        p: pulp.LpVariable(f"Player_{i}", cat='Binary')
        for i, p in enumerate(player_names)
    }
    
    # Create quick-lookup dictionaries for optimization properties to speed up LP equation building
    fantasy_dict = dict(zip(df['Player Name'], df['Fantasy_Score']))
    cost_dict = dict(zip(df['Player Name'], df['Cost_Cr']))
    
    # Override baseline costs with any provided retention/custom auction prices
    for player, custom_price in retention_prices.items():
        if player in cost_dict:
            cost_dict[player] = custom_price
    for player, custom_price in custom_auction_prices.items():
        if player in cost_dict:
            cost_dict[player] = custom_price        

    foreign_dict = dict(zip(df['Player Name'], df['Is_Foreign']))
    subrole_dict = dict(zip(df['Player Name'], df['Subrole']))
    role_dict = dict(zip(df['Player Name'], df['Role']))
    
    # Define the Objective Function: Maximize the sum of (Player Binary Var * Player Fantasy Score)
    prob += pulp.lpSum([fantasy_dict[p] * player_vars[p] for p in player_names]), "Total_Fantasy"
    
    # CORE CONSTRAINTS
    # 1. Total players selected must be exactly 11
    prob += pulp.lpSum([player_vars[p] for p in player_names]) == 11, "Must_Have_Exactly_11_Players"
    # 2. Total cost of selected players must be less than or equal to the defined budget
    prob += pulp.lpSum([cost_dict[p] * player_vars[p] for p in player_names]) <= budget, "Stay_Under_Budget"
    # 3. Selected foreign players cannot exceed the specified maximum
    prob += pulp.lpSum([foreign_dict[p] * player_vars[p] for p in player_names]) <= max_foreigners, "Max_4_Foreigners"

    # COMPOSITION CONSTRAINTS
    if constraint_mode == "Strict (Custom Roles)":
        # Strict constraints: User dictates the exact number of players per role
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'Batsman']) == num_batters, "Exact_Batsmen"
        prob += pulp.lpSum([player_vars[p] for p in player_names if subrole_dict.get(p) == 'WK']) >= 1, "Min_1_WicketKeeper"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'Bowler']) == num_bowlers, "Exact_Bowlers"
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) == 'All-Rounder']) == num_all_rounders, "Exact_AllRounders"

        # Minimum pace/spin distribution
        prob += pulp.lpSum([player_vars[p] for p in player_names if is_pacer(subrole_dict.get(p))]) >= num_pacers, "Min_Pacers"
        prob += pulp.lpSum([player_vars[p] for p in player_names if is_spinner(subrole_dict.get(p))]) >= num_spinners, "Min_Spinners"

    else:
        # Flexible constraints: Ensures a balanced team without hard-coding role counts
        prob += pulp.lpSum([player_vars[p] for p in player_names if subrole_dict.get(p) == 'WK']) >= 1, "Min_1_Wicketkeeper"
        # Ensure at least 5 players can bowl (Bowlers + All-Rounders)
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) in ['Bowler', 'All-Rounder']]) >= 5, "Min_5_Bowling_Options"
        # Ensure at least 6 players can bat (Batsmen + All-Rounders)
        prob += pulp.lpSum([player_vars[p] for p in player_names if role_dict.get(p) in ['Batsman', 'All-Rounder']]) >= 6, "Min_6_Batting_Options"

    # FORCED INCLUSIONS/EXCLUSIONS
    # Force the LP solver to set the binary variable to 1 for required players
    for player in must_include:
        if player in player_vars:
            constraint_name = f"Force_Include_{player.replace(' ', '_').replace('-', '_')}"
            prob += player_vars[player] == 1, constraint_name

    # Force the LP solver to set the binary variable to 0 for banned players
    for player in must_exclude:
        if player in player_vars:
            constraint_name = f"Force_Exclude_{player.replace(' ', '_').replace('-', '_')}"
            prob += player_vars[player] == 0, constraint_name

    # PHASE SPECIALIST CONSTRAINTS
    # For each active specialist requirement, sum the binary vars of qualifying players
    # and enforce that at least one must be selected.
    phase_constraints = [
        (require_pp_bat_specialist,   'pp_bat_specialist',   "Min_1_PP_Bat_Specialist"),
        (require_death_bat_specialist,'death_bat_specialist', "Min_1_Death_Bat_Specialist"),
        (require_pp_bowl_specialist,  'pp_bowl_specialist',  "Min_1_PP_Bowl_Specialist"),
        (require_death_bowl_specialist,'death_bowl_specialist',"Min_1_Death_Bowl_Specialist"),
    ]
    for active, col, constraint_name in phase_constraints:
        if active and col in df.columns:
            # Players flagged as specialists for this phase
            specialist_players = df[df[col] == True]['Player Name'].tolist()
            eligible = [p for p in specialist_players if p in player_vars]
            if eligible:
                prob += (
                    pulp.lpSum([player_vars[p] for p in eligible]) >= 1,
                    constraint_name,
                )
            else:
                print(f"WARNING: {constraint_name} requested but no eligible specialists found in pool — constraint skipped.")
            
    # Execute the LP optimization calculation.
    # _get_solver() picks the best available solver on this machine (system CBC > bundled CBC > GLPK).
    try:
        prob.solve(_get_solver())
    except RuntimeError as e:
        print(f"CRITICAL: Solver not available — {e}")
        return None, 0
    
    # Check if a valid mathematical solution exists given the constraints
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("CRITICAL: Could not find an optimal solution. Your constraints might be too strict.")
        return None, 0

    # Extract the names of the 11 players whose binary variable evaluated to 1
    selected = [p for p in player_names if player_vars[p].varValue == 1]

    # Create a clean DataFrame containing only the selected players and key columns, sorted by Fantasy Score
    # Include Phase_Score, specialist tags, and metadata (price_year, last_season, seasons_played)
    base_cols = ['Player Name', 'Role', 'Subrole', 'Cost_Cr', 'Is_Foreign', 'Fantasy_Score', 'Phase_Score', 'Value_Score']
    meta_cols  = ['price_year', 'last_season', 'seasons_played']
    extra_cols = ['pp_bat_specialist', 'death_bat_specialist', 'pp_bowl_specialist', 'death_bowl_specialist']
    output_cols = [c for c in base_cols + meta_cols + extra_cols if c in df.columns]
    final_team_df = df[df['Player Name'].isin(selected)][output_cols].sort_values(
        by='Fantasy_Score', ascending=False
    ).copy()
    
    # Return the selected team DataFrame and the final total Fantasy Score
    return final_team_df, prob.objective.value()


def get_efficient_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the efficient frontier of the player pool — the Pareto-optimal set of
    players where no other player offers both a higher Fantasy Score AND a lower price.

    A player is on the efficient frontier if there is no other player who simultaneously
    has a higher (or equal) Fantasy Score and a lower (or equal) cost. These are the
    players that represent the best possible tradeoff at every price point — the
    'market-efficient' selections from a Moneyball perspective.

    Args:
        df (pd.DataFrame): Scored player DataFrame containing 'Fantasy_Score' and 'Cost_Cr'.

    Returns:
        pd.DataFrame: Subset of df containing only efficient frontier players,
                      sorted by Cost_Cr ascending for plotting.
    """
    frontier_mask = []
    for _, row in df.iterrows():
        # A player is dominated if any other player has >= Fantasy Score AND <= Cost
        is_dominated = df[
            (df['Fantasy_Score'] >= row['Fantasy_Score']) &
            (df['Cost_Cr'] <= row['Cost_Cr']) &
            (df['Player Name'] != row['Player Name'])
        ]
        frontier_mask.append(is_dominated.empty)

    frontier_df = df[frontier_mask].sort_values('Cost_Cr').copy()
    return frontier_df