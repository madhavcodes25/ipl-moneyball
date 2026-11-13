"""
Phase Specialist Stats Module — IPL Moneyball
==============================================
Computes powerplay / middle-overs / death-overs batting and bowling statistics
from the RR Hackathon ball-by-ball dataset covering IPL 2022–2026.

Each player's per-season phase stats are combined using recency-weighted
averages (older seasons contribute less) so that recent form matters more.

Bowling Composite (per phase)
-----------------------------
  0.5 × norm(phase_wickets)   ← proven wicket-takers rank highest
  0.5 × norm(inv_phase_econ)  ← containment matters equally

``wkts_per_over`` was deliberately dropped: it is derived from the same
numerator/denominator as economy + raw wickets and inflates small-sample
players (e.g. 2 wickets in 12 balls = 1.0 wkts/over vs Bhuvi's 0.3).

Batting Composite (per phase)
------------------------------
  0.4 × norm(phase_SR)   ← scoring rate / aggression
  0.3 × norm(phase_runs) ← volume / consistency
  0.3 × norm(phase_avg)  ← quality, penalises cheap dismissals

Usage
-----
    from src.phase_stats import load_phase_stats
    df = load_phase_stats()          # returns one row per player
    print(df.columns.tolist())

Pakistani Players
-----------------
A hardcoded exclusion list is applied.  As of IPL 2022-2026 no Pakistani
players competed, but the filter is applied defensively.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPL_COMP_IDS = {1358, 1715, 2193, 2725, 3298}   # 2022, 2023, 2024, 2025, 2026

# More recent seasons carry more weight so recent form dominates
SEASON_WEIGHTS: dict = {
    "2022": 1,
    "2023": 2,
    "2024": 3,
    "2025": 4,
    "2026": 5,
}

# Phase definitions (over numbers are 1-indexed, overid in data is 0-indexed 0-19)
# overid  0- 5  → Powerplay  (overs 1-6)
# overid  6-14  → Middle     (overs 7-15)
# overid 15-19  → Death      (overs 16-20)
PHASE_BINS   = [-1, 5, 14, 19]          # right-inclusive with pd.cut
PHASE_LABELS = ["Powerplay", "Middle", "Death"]

# Minimum ball thresholds — below these the sample is too small to trust
MIN_BAT_BALLS: dict = {"Powerplay": 30, "Middle": 20, "Death": 15}
MIN_BOWL_BALLS: dict = {"Powerplay": 24, "Middle": 30, "Death": 24}

# Pakistani player exclusion list (applied defensively)
PAKISTANI_PLAYERS = {
    "Shaheen Afridi", "Shaheen Shah Afridi", "Babar Azam", "Mohammad Rizwan",
    "Naseem Shah", "Haris Rauf", "Shadab Khan", "Mohammad Nawaz",
    "Fakhar Zaman", "Iftikhar Ahmed", "Mohammad Amir", "Shoaib Malik",
    "Sarfaraz Ahmed", "Imad Wasim", "Wahab Riaz", "Mohammad Hafeez",
    "Kamran Akmal", "Umar Akmal",
}

# Default data path (relative to repo root)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "RR-Hackathon-Data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ipl_bbb(data_dir: Path = _DATA_DIR) -> pd.DataFrame:
    """
    Loads the ball-by-ball CSV, filters to IPL competitions only, adds phase
    and boolean helper columns, and returns a clean DataFrame.

    Args:
        data_dir: Path to the RR-Hackathon-Data directory.

    Returns:
        pd.DataFrame with IPL deliveries only plus phase / helper columns.
    """
    bbb_path = data_dir / "ball_by_ball.csv"
    matches_path = data_dir / "matches.csv"

    bbb = pd.read_csv(bbb_path, low_memory=False)
    matches = pd.read_csv(matches_path, usecols=["match_id", "comp_id"])

    # Merge to get comp_id per ball
    bbb = bbb.merge(matches, on="match_id", how="left")

    # Filter IPL competitions
    ipl = bbb[bbb["comp_id"].isin(IPL_COMP_IDS)].copy()

    # --- Coerce boolean-ish columns to numeric ---
    for col in ["is_four", "is_six", "is_wicket"]:
        ipl[col] = pd.to_numeric(
            ipl[col].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}),
            errors="coerce",
        ).fillna(0).astype(int)

    # Wide delivery flag
    ipl["is_wide"] = ipl["extras_type"].str.lower().str.strip() == "wide"

    # Phase label (based on 0-indexed overid: 0-5 PP, 6-14 Middle, 15-19 Death)
    ipl["phase"] = pd.cut(
        ipl["overid"],
        bins=PHASE_BINS,
        labels=PHASE_LABELS,
    )

    # Season string from comp_id
    ipl["season"] = ipl["comp_id"].map(
        {1358: "2022", 1715: "2023", 2193: "2024", 2725: "2025", 3298: "2026"}
    )

    # Remove Pakistani players
    ipl = ipl[~ipl["striker_name"].isin(PAKISTANI_PLAYERS)]
    ipl = ipl[~ipl["bowler_name"].isin(PAKISTANI_PLAYERS)]

    return ipl


def _weighted_avg(group: pd.DataFrame, stat_col: str, balls_col: str,
                  weight_col: str) -> float:
    """
    Computes a ball-weighted, recency-weighted average of a rate stat
    across seasons.

    Formula:
        Σ (balls_s × stat_s × weight_s)  /  Σ (balls_s × weight_s)

    Args:
        group: DataFrame slice for one player+phase.
        stat_col: Column holding the per-season rate (e.g. strike_rate).
        balls_col: Column holding balls faced/bowled in that season.
        weight_col: Column holding the season weight.

    Returns:
        Weighted average float, or NaN if denominator is zero.
    """
    num = (group[balls_col] * group[stat_col] * group[weight_col]).sum()
    den = (group[balls_col] * group[weight_col]).sum()
    return num / den if den > 0 else np.nan


# ---------------------------------------------------------------------------
# Phase batting stats
# ---------------------------------------------------------------------------

def _compute_batting_phase_stats(ipl: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-player, per-phase, per-season batting stats then collapses
    across seasons using recency-weighted averages.

    Stats computed
    --------------
    - Strike Rate (SR): runs per 100 legal balls — scoring rate / aggression
    - Runs: total runs in that phase — volume / consistency metric
    - Average (avg): runs per dismissal — quality, penalises cheap dismissals
      (computed as season-avg weighted by balls; 0 dismissals → avg = runs for that season)

    Args:
        ipl: Filtered IPL ball-by-ball DataFrame with phase & season columns.

    Returns:
        Wide DataFrame indexed by player_name with columns:
        pp_bat_sr,   middle_bat_sr,   death_bat_sr,
        pp_bat_runs, middle_bat_runs,  death_bat_runs,
        pp_bat_avg,  middle_bat_avg,   death_bat_avg,
        pp_bat_balls, middle_bat_balls, death_bat_balls
    """
    # Legal deliveries only (exclude wides from ball/run count)
    legal = ipl[~ipl["is_wide"]].copy()

    agg = (
        legal.groupby(
            ["striker_name", "season", "phase"], observed=True
        )
        .agg(
            balls=("runs", "count"),
            runs=("runs", "sum"),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
    )

    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    # Average = runs / dismissals; 0 dismissals treated as not-out → use runs as proxy
    agg["bat_avg"] = (agg["runs"] / agg["dismissals"].replace(0, np.nan)).fillna(agg["runs"]).round(2)
    agg["weight"]  = agg["season"].map(SEASON_WEIGHTS).fillna(1)

    records = []
    for (player, phase), grp in agg.groupby(["striker_name", "phase"], observed=True):
        w_sr    = _weighted_avg(grp, "strike_rate", "balls", "weight")
        w_avg   = _weighted_avg(grp, "bat_avg",     "balls", "weight")
        total_balls = grp["balls"].sum()
        # Weighted runs: sum of (runs_s × weight_s) / sum(weight_s) — a normalised volume metric
        total_weight = grp["weight"].sum()
        w_runs = (grp["runs"] * grp["weight"]).sum() / total_weight if total_weight > 0 else grp["runs"].sum()
        records.append({
            "player_name": player,
            "phase":       phase,
            "bat_sr":      w_sr,
            "bat_runs":    round(w_runs, 2),
            "bat_avg":     w_avg,
            "bat_balls":   total_balls,
        })

    phase_df = pd.DataFrame(records)

    # Apply minimum ball thresholds — null all rate stats below threshold
    for phase_name, min_b in MIN_BAT_BALLS.items():
        mask = (phase_df["phase"] == phase_name) & (phase_df["bat_balls"] < min_b)
        phase_df.loc[mask, ["bat_sr", "bat_runs", "bat_avg"]] = np.nan

    # Pivot to wide format
    wide = phase_df.pivot_table(
        index="player_name",
        columns="phase",
        values=["bat_sr", "bat_runs", "bat_avg", "bat_balls"],
        aggfunc="first",
    )
    wide.columns = [f"{col[1].lower()}_{col[0]}" for col in wide.columns]
    wide = wide.rename(columns={
        "powerplay_bat_sr":    "pp_bat_sr",
        "middle_bat_sr":       "middle_bat_sr",
        "death_bat_sr":        "death_bat_sr",
        "powerplay_bat_runs":  "pp_bat_runs",
        "middle_bat_runs":     "middle_bat_runs",
        "death_bat_runs":      "death_bat_runs",
        "powerplay_bat_avg":   "pp_bat_avg",
        "middle_bat_avg":      "middle_bat_avg",
        "death_bat_avg":       "death_bat_avg",
        "powerplay_bat_balls": "pp_bat_balls",
        "middle_bat_balls":    "middle_bat_balls",
        "death_bat_balls":     "death_bat_balls",
    })
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Phase bowling stats
# ---------------------------------------------------------------------------

def _compute_bowling_phase_stats(ipl: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-player, per-phase, per-season bowling stats then collapses
    across seasons using recency-weighted averages.

    Stats computed (phase-specific)
    --------------------------------
    - Economy (bowl_econ): runs conceded per 6 balls — measures containment
    - Wickets (bowl_wkts): total bowler-attributed wickets in that phase —
      the primary measure of wicket-taking ability (volume, not just rate).

    Note: wkts_per_over (rate metric) was removed because it is derivable
    from wickets and balls and inflates small-sample players who happened to
    get lucky in a handful of deliveries.  Raw phase wickets + economy gives
    a cleaner, non-redundant signal.

    Economy is computed on all deliveries (wides count as balls + runs);
    wickets exclude run-outs and 'obstructing the field' (attributed to fielder).

    Args:
        ipl: Filtered IPL ball-by-ball DataFrame with phase & season columns.

    Returns:
        Wide DataFrame indexed by player_name with columns:
        pp_bowl_econ,   middle_bowl_econ,   death_bowl_econ,
        pp_bowl_wkts,   middle_bowl_wkts,   death_bowl_wkts,
        pp_bowl_balls,  middle_bowl_balls,  death_bowl_balls
    """
    bowl = ipl.copy()

    # Runs conceded = runs on the delivery (extras included but not bye/leg-bye)
    bowl["runs_conceded"] = bowl["runs"].copy()

    # Wickets attributed to bowler (exclude run-outs, obstructing the field)
    runout_types = {"run out", "runout", "obstructing the field"}
    bowl["bowl_wicket"] = np.where(
        bowl["is_wicket"] == 1,
        ~bowl["dismissal_type"].str.lower().str.strip().isin(runout_types),
        False,
    ).astype(int)

    # All deliveries count for economy (including wides — they cost runs & a ball)
    agg = (
        bowl.groupby(
            ["bowler_name", "season", "phase"], observed=True
        )
        .agg(
            balls=("runs_conceded", "count"),
            runs_conceded=("runs_conceded", "sum"),
            wickets=("bowl_wicket", "sum"),
        )
        .reset_index()
    )

    # Economy = runs per 6 balls
    agg["economy"] = ((agg["runs_conceded"] / agg["balls"]) * 6).round(3)
    agg["weight"]  = agg["season"].map(SEASON_WEIGHTS).fillna(1)

    records = []
    for (player, phase), grp in agg.groupby(["bowler_name", "phase"], observed=True):
        w_econ      = _weighted_avg(grp, "economy", "balls", "weight")
        total_balls = grp["balls"].sum()
        total_wkts  = grp["wickets"].sum()   # raw phase wickets — simple sum across seasons
        records.append({
            "player_name":  player,
            "phase":        phase,
            "bowl_econ":    w_econ,
            "bowl_wickets": total_wkts,       # phase-specific wicket count
            "bowl_balls":   total_balls,
        })

    phase_df = pd.DataFrame(records)

    # Apply minimum ball thresholds — null stats below threshold
    for phase_name, min_b in MIN_BOWL_BALLS.items():
        mask = (phase_df["phase"] == phase_name) & (phase_df["bowl_balls"] < min_b)
        phase_df.loc[mask, ["bowl_econ", "bowl_wickets"]] = np.nan

    # Pivot to wide format
    wide = phase_df.pivot_table(
        index="player_name",
        columns="phase",
        values=["bowl_econ", "bowl_wickets", "bowl_balls"],
        aggfunc="first",
    )
    wide.columns = [f"{col[1].lower()}_{col[0]}" for col in wide.columns]
    wide = wide.rename(columns={
        "powerplay_bowl_econ":    "pp_bowl_econ",
        "middle_bowl_econ":       "middle_bowl_econ",
        "death_bowl_econ":        "death_bowl_econ",
        "powerplay_bowl_wickets": "pp_bowl_wkts",
        "middle_bowl_wickets":    "middle_bowl_wkts",
        "death_bowl_wickets":     "death_bowl_wkts",
        "powerplay_bowl_balls":   "pp_bowl_balls",
        "middle_bowl_balls":      "middle_bowl_balls",
        "death_bowl_balls":       "death_bowl_balls",
    })
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_phase_stats(data_dir: Path = _DATA_DIR) -> pd.DataFrame:
    """
    Main entry point.  Loads, processes, and returns a tidy per-player
    phase stats DataFrame that can be merged with the player pool.

    Columns returned
    ----------------
    player_name,
    pp_bat_sr,     middle_bat_sr,     death_bat_sr,       # phase strike rates
    pp_bat_runs,   middle_bat_runs,   death_bat_runs,     # phase runs (weighted)
    pp_bat_avg,    middle_bat_avg,    death_bat_avg,      # phase batting averages
    pp_bat_balls,  middle_bat_balls,  death_bat_balls,    # balls faced (quality gate)
    pp_bowl_econ,  middle_bowl_econ,  death_bowl_econ,    # phase economies
    pp_bowl_wkts,  middle_bowl_wkts,  death_bowl_wkts,   # phase wickets taken
    pp_bowl_balls, middle_bowl_balls, death_bowl_balls    # balls bowled (quality gate)

    Args:
        data_dir: Path to RR-Hackathon-Data directory.
                  Defaults to data/RR-Hackathon-Data relative to repo root.

    Returns:
        pd.DataFrame with one row per player (players with no data in a phase
        will have NaN for that phase's stats).
    """
    print("[phase_stats] Loading IPL ball-by-ball data …")
    ipl = _load_ipl_bbb(data_dir)
    print(f"[phase_stats] {len(ipl):,} IPL deliveries loaded across {ipl['season'].nunique()} seasons")

    bat_df = _compute_batting_phase_stats(ipl)
    bowl_df = _compute_bowling_phase_stats(ipl)

    merged = bat_df.merge(bowl_df, on="player_name", how="outer")

    # Remove Pakistani players from final output (defensive second pass)
    merged = merged[~merged["player_name"].isin(PAKISTANI_PLAYERS)].copy()

    print(f"[phase_stats] Phase stats computed for {len(merged):,} players")
    return merged
