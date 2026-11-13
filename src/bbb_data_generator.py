"""
BBB Data Generator — IPL Moneyball
====================================
Generates a comprehensive player stats CSV derived entirely from the RR Hackathon
ball-by-ball dataset (IPL 2022-2026).  This replaces the old hand-curated
IPL_Stats.csv as the primary player pool, adding 100+ new players, multi-year
recency-weighted stats, and richer metadata (bowling type → role/subrole, WK flag).

Usage
-----
    python -m src.bbb_data_generator            # writes data/processed/bbb_derived_stats.csv
    from src.bbb_data_generator import generate_bbb_stats
    df = generate_bbb_stats()

Output columns
--------------
Player Name, Role, Subrole, Is_Foreign,
Batting Runs, Batting Average, Batting Strike Rate,
Wickets, Economy Rate, Bowling Average,
Cost_Cr,                       ← carried over from IPL_Stats.csv where known; else median
seasons_played,                ← number of IPL seasons the player appeared
last_season                    ← most recent IPL season the player appeared in
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPL_COMP_IDS = {1358, 1715, 2193, 2725, 3298}
SEASON_MAP   = {1358: "2022", 1715: "2023", 2193: "2024", 2725: "2025", 3298: "2026"}
SEASON_WEIGHTS = {"2022": 1, "2023": 2, "2024": 3, "2025": 4, "2026": 5}

PAKISTANI_PLAYERS = {
    "Shaheen Afridi", "Shaheen Shah Afridi", "Babar Azam", "Mohammad Rizwan",
    "Naseem Shah", "Haris Rauf", "Shadab Khan", "Mohammad Nawaz",
    "Fakhar Zaman", "Iftikhar Ahmed", "Mohammad Amir", "Shoaib Malik",
    "Sarfaraz Ahmed", "Imad Wasim", "Wahab Riaz", "Mohammad Hafeez",
    "Kamran Akmal", "Umar Akmal",
}

# Bowling-type → Subrole mapping
SPIN_TYPES = {
    "right-arm leg-break", "right-arm offbreak", "slow left-arm orthodox",
    "left-arm chinaman", "right-arm leg-spin", "right-arm off-break",
    "slow left-arm", "right-arm slow",
}
PACE_TYPES = {
    "right-arm fast", "right-arm fast-medium", "right-arm medium",
    "right-arm medium-fast", "left-arm fast", "left-arm fast-medium",
    "left-arm medium", "left-arm medium-fast",
}

# Minimum thresholds to be considered "has batting / bowling data"
MIN_BAT_BALLS  = 30   # at least 30 legal balls faced across career
MIN_BOWL_BALLS = 60   # at least 60 deliveries bowled across career (10 overs)

_DATA_DIR    = Path(__file__).resolve().parent.parent / "data" / "RR-Hackathon-Data"
_OLD_CSV     = Path(__file__).resolve().parent.parent / "data" / "raw" / "IPL_Stats.csv"
_PRICE_2026  = Path(__file__).resolve().parent.parent / "data" / "raw" / "IPL_2026_prices.csv"
_OUT_CSV     = Path(__file__).resolve().parent.parent / "data" / "processed" / "bbb_derived_stats.csv"

# Name aliases: BBB full names → IPL_Stats.csv names (for cost / Is_Foreign lookup)
BBB_TO_OLD_NAME: dict = {
    "Mahendra Singh Dhoni":  "MS Dhoni",
    "Suryakumar Yadav":      "Surya Kumar Yadav",
    "Lokesh Rahul":          "K L Rahul",
    "Atharwa Taide":         "Atharva Taide",
    "Namboori Tilak Varma":  "Tilak Varma",
    "Mukul Choudhary":       "Mukesh Choudhary",
    "Nishant Kumar Sindhu":  "Nishant Sindhu",
    "Faf du Plessis":        "Faf Du Plessis",
    "Nitish Reddy":          "Nitish Kumar Reddy",
    "Himmat Singh":          "Himmat Singh Deol",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wavg(grp: pd.DataFrame, stat: str, balls_col: str = "balls",
          w_col: str = "weight") -> float:
    """Ball-weighted, recency-weighted average of a rate stat across seasons."""
    num = (grp[balls_col] * grp[stat] * grp[w_col]).sum()
    den = (grp[balls_col] * grp[w_col]).sum()
    return num / den if den > 0 else np.nan


def _infer_subrole(bowling_type, is_wk: bool) -> str:
    """Infer Subrole from bowling type and WK flag."""
    if is_wk:
        return "WK"
    if pd.isna(bowling_type) or bowling_type == "":
        return "Non-WK"
    bt = str(bowling_type).lower().strip()
    if bt in SPIN_TYPES:
        return "Spinner"
    if bt in PACE_TYPES:
        return "Pacer"
    # Partial match fallback
    if any(k in bt for k in ["spin", "orthodox", "leg-break", "off-break", "chinaman", "slow"]):
        return "Spinner"
    if any(k in bt for k in ["fast", "medium", "seam", "pace"]):
        return "Pacer"
    return "Non-WK"


def _infer_role(has_bat: bool, has_bowl: bool, subrole: str) -> str:
    """Infer primary Role from batting/bowling activity."""
    if has_bat and has_bowl:
        return "All-Rounder"
    if has_bowl:
        return "Bowler"
    return "Batsman"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_bbb_stats(
    data_dir: Path = _DATA_DIR,
    old_csv: Path = _OLD_CSV,
    out_csv: Path = _OUT_CSV,
) -> pd.DataFrame:
    """
    Derive a full player statistics table from IPL ball-by-ball data.

    Steps
    -----
    1. Load BBB data, filter to IPL 2022-2026, clean booleans.
    2. Identify WK players from playing_eleven.csv.
    3. Get each player's bowling type from BBB.
    4. Compute career (multi-year recency-weighted) batting and bowling stats.
    5. Assign Role / Subrole from activity and bowling type.
    6. Bring in Cost_Cr / Is_Foreign from old IPL_Stats.csv where available.
    7. Write to data/processed/bbb_derived_stats.csv and return DataFrame.

    Args:
        data_dir: Path to RR-Hackathon-Data directory.
        old_csv: Path to legacy IPL_Stats.csv (for cost / foreign flag fallback).
        out_csv: Output path for the generated CSV.

    Returns:
        pd.DataFrame with one row per player.
    """
    print("[bbb_generator] Loading ball-by-ball data …")
    bbb     = pd.read_csv(data_dir / "ball_by_ball.csv", low_memory=False)
    matches = pd.read_csv(data_dir / "matches.csv", usecols=["match_id", "comp_id"])
    p11     = pd.read_csv(data_dir / "playing_eleven.csv")

    # Merge comp_id into bbb and filter IPL
    bbb  = bbb.merge(matches, on="match_id", how="left")
    ipl  = bbb[bbb["comp_id"].isin(IPL_COMP_IDS)].copy()
    ipl["season"] = ipl["comp_id"].map(SEASON_MAP)

    # Remove Pakistani players
    ipl = ipl[~ipl["striker_name"].isin(PAKISTANI_PLAYERS)].copy()
    ipl = ipl[~ipl["bowler_name"].isin(PAKISTANI_PLAYERS)].copy()

    # Coerce boolean columns
    for col in ["is_four", "is_six", "is_wicket"]:
        ipl[col] = pd.to_numeric(
            ipl[col].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}),
            errors="coerce",
        ).fillna(0).astype(int)
    ipl["is_wide"] = ipl["extras_type"].str.lower().str.strip() == "wide"

    # -----------------------------------------------------------------------
    # WK identification
    # -----------------------------------------------------------------------
    ipl_match_ids = set(matches[matches["comp_id"].isin(IPL_COMP_IDS)]["match_id"])
    ipl_p11 = p11[p11["match_id"].isin(ipl_match_ids)].copy()
    ipl_p11["is_wk"] = ipl_p11["is_wk"].map(
        {True: True, False: False, "true": True, "false": False, 1: True, 0: False}
    ).fillna(False)
    wk_ids = set(ipl_p11[ipl_p11["is_wk"] == True]["player_id"].unique())

    # Build player_id → player_name map
    id2name = pd.concat([
        ipl[["striker_id", "striker_name"]].rename(columns={"striker_id": "player_id", "striker_name": "player_name"}),
        ipl[["bowler_id", "bowler_name"]].rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"}),
    ]).drop_duplicates("player_id")
    wk_names = set(id2name[id2name["player_id"].isin(wk_ids)]["player_name"])

    # -----------------------------------------------------------------------
    # Bowling type per player
    # -----------------------------------------------------------------------
    bowl_type_map = (
        ipl[["bowler_name", "bowling_type"]]
        .drop_duplicates("bowler_name")
        .set_index("bowler_name")["bowling_type"]
        .to_dict()
    )

    # -----------------------------------------------------------------------
    # Batting stats (recency-weighted across seasons)
    # -----------------------------------------------------------------------
    legal = ipl[~ipl["is_wide"]].copy()
    bat_season = (
        legal.groupby(["striker_name", "season"])
        .agg(
            balls=("runs", "count"),
            runs=("runs", "sum"),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
    )
    bat_season["weight"] = bat_season["season"].map(SEASON_WEIGHTS).fillna(1)
    bat_season["sr"]  = bat_season["runs"] / bat_season["balls"] * 100
    bat_season["avg"] = (bat_season["runs"] / bat_season["dismissals"].replace(0, np.nan)).fillna(bat_season["runs"])

    bat_records = []
    for name, grp in bat_season.groupby("striker_name"):
        total_balls = grp["balls"].sum()
        total_wt    = grp["weight"].sum()
        w_runs = (grp["runs"] * grp["weight"]).sum() / total_wt if total_wt > 0 else 0
        w_sr   = _wavg(grp, "sr")
        w_avg  = _wavg(grp, "avg")
        seasons_bat = grp["season"].nunique()
        last_bat    = grp["season"].max()
        bat_records.append({
            "Player Name":           name,
            "total_bat_balls":       int(total_balls),
            "Batting Runs":          round(w_runs, 1),
            "Batting Strike Rate":   round(w_sr, 2)  if pd.notna(w_sr)  else 0.0,
            "Batting Average":       round(w_avg, 2) if pd.notna(w_avg) else 0.0,
            "seasons_bat":           seasons_bat,
            "last_bat_season":       last_bat,
        })
    bat_df = pd.DataFrame(bat_records)

    # -----------------------------------------------------------------------
    # Bowling stats (recency-weighted across seasons)
    # -----------------------------------------------------------------------
    runout_types = {"run out", "runout", "obstructing the field"}
    ipl["bowl_wicket"] = np.where(
        ipl["is_wicket"] == 1,
        ~ipl["dismissal_type"].str.lower().str.strip().isin(runout_types),
        False,
    ).astype(int)

    bowl_season = (
        ipl.groupby(["bowler_name", "season"])
        .agg(
            balls=("runs", "count"),
            runs=("runs", "sum"),
            wickets=("bowl_wicket", "sum"),
        )
        .reset_index()
    )
    bowl_season["weight"] = bowl_season["season"].map(SEASON_WEIGHTS).fillna(1)
    bowl_season["econ"]   = bowl_season["runs"] / bowl_season["balls"] * 6

    bowl_records = []
    for name, grp in bowl_season.groupby("bowler_name"):
        total_balls = grp["balls"].sum()
        total_wkts  = grp["wickets"].sum()
        total_wt    = grp["weight"].sum()
        w_econ = _wavg(grp, "econ")
        # Bowling average: weighted runs per wicket
        w_runs = (grp["runs"] * grp["weight"]).sum() / total_wt if total_wt > 0 else 0
        w_bowl_avg = round(w_runs / total_wkts, 2) if total_wkts > 0 else 0.0
        seasons_bowl = grp["season"].nunique()
        last_bowl    = grp["season"].max()
        bowl_records.append({
            "Player Name":     name,
            "total_bowl_balls": int(total_balls),
            "Wickets":         int(total_wkts),
            "Economy Rate":    round(w_econ, 2) if pd.notna(w_econ) else 0.0,
            "Bowling Average": w_bowl_avg,
            "seasons_bowl":    seasons_bowl,
            "last_bowl_season": last_bowl,
        })
    bowl_df = pd.DataFrame(bowl_records)

    # -----------------------------------------------------------------------
    # Merge batting + bowling
    # -----------------------------------------------------------------------
    merged = bat_df.merge(bowl_df, on="Player Name", how="outer")

    # Mark "has" flags based on minimum thresholds
    merged["has_bat"]  = merged["total_bat_balls"].fillna(0)  >= MIN_BAT_BALLS
    merged["has_bowl"] = merged["total_bowl_balls"].fillna(0) >= MIN_BOWL_BALLS

    # Fill NaN stats with 0
    for col in ["Batting Runs", "Batting Strike Rate", "Batting Average",
                "Wickets", "Economy Rate", "Bowling Average"]:
        merged[col] = merged[col].fillna(0)

    # Subrole and Role
    merged["is_wk"]   = merged["Player Name"].isin(wk_names)
    merged["bowl_type"] = merged["Player Name"].map(bowl_type_map)
    merged["Subrole"] = merged.apply(
        lambda r: _infer_subrole(r["bowl_type"], r["is_wk"]), axis=1
    )
    merged["Role"] = merged.apply(
        lambda r: _infer_role(r["has_bat"], r["has_bowl"], r["Subrole"]), axis=1
    )

    # seasons_played / last_season
    merged["seasons_played"] = merged[["seasons_bat", "seasons_bowl"]].max(axis=1).fillna(0).astype(int)
    # last_bat_season and last_bowl_season may be NaN for pure bat/bowl players; coerce to str first
    merged["last_bat_season"]  = merged["last_bat_season"].fillna("2022").astype(str)
    merged["last_bowl_season"] = merged["last_bowl_season"].fillna("2022").astype(str)
    merged["last_season"]      = merged[["last_bat_season", "last_bowl_season"]].max(axis=1)

    # -----------------------------------------------------------------------
    # Cost_Cr, Is_Foreign, and price_year
    # Priority:
    #   1. IPL 2026 prices (data/raw/IPL_2026_prices.csv) — retentions + auction
    #   2. Direct name match in IPL_Stats.csv (legacy, sorted by Year desc)
    #   3. Alias map (BBB_TO_OLD_NAME) for known name mismatches
    #   4. Role+Subrole-based median of known prices
    # -----------------------------------------------------------------------
    # Build 2026 price lookup (Player Name → {Cost_Cr, Is_Foreign})
    price_2026_map: dict = {}
    if _PRICE_2026.exists():
        p26 = pd.read_csv(_PRICE_2026)
        p26 = p26.drop_duplicates("Player Name", keep="first")
        for _, r in p26.iterrows():
            price_2026_map[r["Player Name"]] = {
                "Cost_Cr": float(r["Cost_Cr"]),
                "Is_Foreign": int(r.get("Is_Foreign", 0)),
                "price_year": "2026",
            }

    # Build legacy price lookup from IPL_Stats.csv
    legacy_meta: dict = {}
    if old_csv.exists():
        old_df = pd.read_csv(old_csv)
        old_df = old_df.sort_values("Year", ascending=False).drop_duplicates("Player Name", keep="first")
        for _, r in old_df.iterrows():
            if r["Player Name"] not in legacy_meta:
                legacy_meta[r["Player Name"]] = {
                    "Cost_Cr": float(r["Cost_Cr"]) if pd.notna(r["Cost_Cr"]) else np.nan,
                    "Is_Foreign": int(r.get("Is_Foreign", 0)),
                    "price_year": str(int(r["Year"])) if pd.notna(r.get("Year")) else "legacy",
                }

    def _get_price(bbb_name: str) -> dict:
        """Lookup in 2026 → legacy → alias → empty."""
        # 1. Direct 2026 hit
        if bbb_name in price_2026_map:
            return price_2026_map[bbb_name]
        # 2. Alias → 2026
        alias = BBB_TO_OLD_NAME.get(bbb_name)
        if alias and alias in price_2026_map:
            return price_2026_map[alias]
        # 3. Direct legacy hit
        if bbb_name in legacy_meta:
            return legacy_meta[bbb_name]
        # 4. Alias → legacy
        if alias and alias in legacy_meta:
            return legacy_meta[alias]
        return {}

    merged["Cost_Cr"]    = merged["Player Name"].apply(lambda n: _get_price(n).get("Cost_Cr",    np.nan))
    merged["Is_Foreign"] = merged["Player Name"].apply(lambda n: _get_price(n).get("Is_Foreign", 0))
    merged["price_year"] = merged["Player Name"].apply(lambda n: _get_price(n).get("price_year", None))

    # Role+Subrole-based median fallback for players with no known price
    role_subrole_medians = (
        merged[merged["Cost_Cr"].notna()]
        .groupby(["Role", "Subrole"])["Cost_Cr"]
        .median()
    )
    global_median = merged["Cost_Cr"].median()
    if pd.isna(global_median):
        global_median = 2.0

    def _fill_cost(row):
        if pd.notna(row["Cost_Cr"]):
            return row["Cost_Cr"]
        key = (row["Role"], row["Subrole"])
        if key in role_subrole_medians.index:
            return role_subrole_medians[key]
        return global_median

    merged["Cost_Cr"]    = merged.apply(_fill_cost, axis=1)
    merged["price_year"] = merged["price_year"].fillna("estimate")
    merged["Is_Foreign"] = merged["Is_Foreign"].fillna(0).astype(int)

    # -----------------------------------------------------------------------
    # Final clean-up
    # -----------------------------------------------------------------------
    # Apply minimum-ball quality gates — zero out stats below threshold
    merged.loc[~merged["has_bat"],  ["Batting Runs", "Batting Strike Rate", "Batting Average"]] = 0
    merged.loc[~merged["has_bowl"], ["Wickets", "Economy Rate", "Bowling Average"]] = 0

    # Keep only relevant output columns
    out_cols = [
        "Player Name", "Role", "Subrole", "Is_Foreign", "Cost_Cr", "price_year",
        "Batting Runs", "Batting Average", "Batting Strike Rate",
        "Wickets", "Economy Rate", "Bowling Average",
        "seasons_played", "last_season",
        "total_bat_balls", "total_bowl_balls",
    ]
    out = merged[out_cols].copy()

    # Ensure output directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[bbb_generator] Written {len(out)} players → {out_csv}")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_bbb_stats()
    print(df[["Player Name", "Role", "Subrole", "Batting Runs", "Batting Strike Rate",
              "Wickets", "Economy Rate", "Cost_Cr"]].head(20).to_string())
