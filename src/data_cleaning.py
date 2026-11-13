from pathlib import Path
import pandas as pd
import numpy as np

# Pakistani players excluded defensively (none historically in IPL, but applied as a safety filter)
_PAKISTANI_PLAYERS = {
    "Shaheen Afridi", "Shaheen Shah Afridi", "Babar Azam", "Mohammad Rizwan",
    "Naseem Shah", "Haris Rauf", "Shadab Khan", "Mohammad Nawaz",
    "Fakhar Zaman", "Iftikhar Ahmed", "Mohammad Amir", "Shoaib Malik",
    "Sarfaraz Ahmed", "Imad Wasim", "Wahab Riaz", "Mohammad Hafeez",
    "Kamran Akmal", "Umar Akmal",
}

# Phase stat columns that will be added from the RR ball-by-ball data
PHASE_COLS = [
    "pp_bat_sr", "middle_bat_sr", "death_bat_sr",
    "pp_bat_boundary_pct", "middle_bat_boundary_pct", "death_bat_boundary_pct",
    "pp_bat_balls", "middle_bat_balls", "death_bat_balls",
    "pp_bowl_econ", "middle_bowl_econ", "death_bowl_econ",
    "pp_bowl_wkts_po", "middle_bowl_wkts_po", "death_bowl_wkts_po",
    "pp_bowl_wkts", "middle_bowl_wkts", "death_bowl_wkts",
    "pp_bowl_balls", "middle_bowl_balls", "death_bowl_balls",
]


def prepare_data(file1=None):
    """
    Loads and preprocesses IPL player statistics from a CSV file.

    This function reads the raw data, ensures the 'Year' column is numeric,
    sorts the data to prioritize the most recent records, and removes duplicate
    player entries to keep only their latest statistics. It also enforces
    numeric data types for all relevant statistical and metadata columns.

    Phase specialist stats (Powerplay / Middle / Death) derived from the
    RR Hackathon ball-by-ball dataset (IPL 2022–2026) are merged in via
    ``src.phase_stats.load_phase_stats()``.  Players with no ball-by-ball data
    get NaN for the phase columns (all downstream code handles NaN gracefully).

    Args:
        file1 (str or Path, optional): Filepath to the raw IPL statistics CSV.
                                      Defaults to data/raw/IPL_Stats.csv.

    Returns:
        pd.DataFrame: A cleaned and deduplicated DataFrame ready for analysis,
                      including phase specialist columns.
    """
    if file1 is None:
        default_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "IPL_Stats.csv"
        file1 = default_path if default_path.exists() else "data/raw/IPL_Stats.csv"

    print("Loading data...")
    df = pd.read_csv(file1)

    # Exclude Pakistani players (defensive filter)
    df = df[~df["Player Name"].isin(_PAKISTANI_PLAYERS)].copy()

    # Convert the 'Year' column to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Sort descending so the most recent season is first
    df = df.sort_values(by="Year", ascending=False)

    # Keep only the most recent season per player
    df = df.drop_duplicates(subset=["Player Name"], keep="first").copy()

    # Core performance metrics — coerce to numeric
    numeric_cols = [
        "Batting Runs", "Batting Average", "Batting Strike Rate",
        "Wickets", "Economy Rate", "Bowling Average",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Financial & flag columns
    df["Cost_Cr"] = pd.to_numeric(df["Cost_Cr"], errors="coerce")
    df["Is_Foreign"] = pd.to_numeric(df["Is_Foreign"], errors="coerce")

    # -----------------------------------------------------------------------
    # Merge phase specialist stats from RR Hackathon ball-by-ball data
    # -----------------------------------------------------------------------
    try:
        from src.phase_stats import load_phase_stats
        phase_df = load_phase_stats()

        # Merge on player name (left join — keep all IPL_Stats players)
        df = df.merge(
            phase_df.rename(columns={"player_name": "Player Name"}),
            on="Player Name",
            how="left",
        )

        # Ensure all phase columns are numeric
        for col in PHASE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        matched = df[PHASE_COLS[0]].notna().sum()
        print(f"[data_cleaning] Phase stats merged: {matched}/{len(df)} players matched")

    except Exception as exc:
        print(f"[data_cleaning] WARNING: Could not load phase stats — {exc}")
        # Populate phase columns with NaN so downstream code doesn't crash
        for col in PHASE_COLS:
            df[col] = np.nan

    return df
