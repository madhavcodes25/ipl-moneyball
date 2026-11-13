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
    "pp_bat_sr",   "middle_bat_sr",   "death_bat_sr",
    "pp_bat_runs", "middle_bat_runs", "death_bat_runs",
    "pp_bat_avg",  "middle_bat_avg",  "death_bat_avg",
    "pp_bat_balls","middle_bat_balls","death_bat_balls",
    "pp_bowl_econ",   "middle_bowl_econ",   "death_bowl_econ",
    "pp_bowl_wkts",   "middle_bowl_wkts",   "death_bowl_wkts",
    "pp_bowl_balls",  "middle_bowl_balls",  "death_bowl_balls",
]

# Paths
_BBB_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "bbb_derived_stats.csv"
_OLD_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "IPL_Stats.csv"


def prepare_data(file1=None):
    """
    Loads and preprocesses IPL player statistics.

    PRIMARY SOURCE: data/processed/bbb_derived_stats.csv (derived from RR Hackathon
    ball-by-ball data, IPL 2022-2026).  This file contains 381 players with
    multi-year recency-weighted stats and richer metadata (bowling type → role,
    WK flag) than the old hand-curated IPL_Stats.csv.

    FALLBACK: If the BBB-derived CSV is not found (or ``file1`` is explicitly
    provided), falls back to the legacy data/raw/IPL_Stats.csv.

    Phase specialist stats (Powerplay / Middle / Death) are merged in from
    ``src.phase_stats.load_phase_stats()``.  Players with no ball-by-ball data
    get NaN for the phase columns (all downstream code handles NaN gracefully).

    Args:
        file1 (str or Path, optional): Explicit path to a player CSV.
                                       Omit to use the BBB-derived CSV (recommended).

    Returns:
        pd.DataFrame: A cleaned and deduplicated DataFrame ready for analysis,
                      including phase specialist columns.
    """
    # -----------------------------------------------------------------------
    # Choose data source
    # -----------------------------------------------------------------------
    if file1 is not None:
        source_path = Path(file1)
        source_label = "custom"
    elif _BBB_CSV.exists():
        source_path = _BBB_CSV
        source_label = "BBB-derived"
    else:
        source_path = _OLD_CSV if _OLD_CSV.exists() else Path("data/raw/IPL_Stats.csv")
        source_label = "legacy IPL_Stats"

    print(f"[data_cleaning] Loading player data from {source_label}: {source_path.name}")
    df = pd.read_csv(source_path)

    # Exclude Pakistani players (defensive filter)
    df = df[~df["Player Name"].isin(_PAKISTANI_PLAYERS)].copy()

    # -----------------------------------------------------------------------
    # Legacy CSV path: deduplicate by most-recent year
    # -----------------------------------------------------------------------
    if source_label != "BBB-derived" and "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.sort_values(by="Year", ascending=False)
        df = df.drop_duplicates(subset=["Player Name"], keep="first").copy()

    # -----------------------------------------------------------------------
    # Ensure required columns exist with correct types
    # -----------------------------------------------------------------------
    numeric_cols = [
        "Batting Runs", "Batting Average", "Batting Strike Rate",
        "Wickets", "Economy Rate", "Bowling Average",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    df["Cost_Cr"]   = pd.to_numeric(df.get("Cost_Cr",   2.0), errors="coerce").fillna(2.0)
    df["Is_Foreign"]= pd.to_numeric(df.get("Is_Foreign", 0),   errors="coerce").fillna(0).astype(int)

    # Role / Subrole defaults if missing
    if "Role" not in df.columns:
        df["Role"] = "Batsman"
    if "Subrole" not in df.columns:
        df["Subrole"] = "Non-WK"

    # -----------------------------------------------------------------------
    # Merge phase specialist stats from RR Hackathon ball-by-ball data
    # (always merged regardless of which base CSV was loaded)
    # -----------------------------------------------------------------------
    try:
        from src.phase_stats import load_phase_stats
        phase_df = load_phase_stats()

        df = df.merge(
            phase_df.rename(columns={"player_name": "Player Name"}),
            on="Player Name",
            how="left",
        )

        for col in PHASE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        matched = df[PHASE_COLS[0]].notna().sum()
        print(f"[data_cleaning] Phase stats merged: {matched}/{len(df)} players matched")

    except Exception as exc:
        print(f"[data_cleaning] WARNING: Could not load phase stats — {exc}")
        for col in PHASE_COLS:
            df[col] = np.nan

    return df
