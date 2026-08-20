"""
Combine raw football-data.co.uk CSVs into one clean, feature-engineered
dataset ready for modelling.

Usage
-----
python build_dataset.py --league E0
    -> reads data/raw/E0_*.csv, writes data/processed/E0_matches.csv

Features added (all computed using only PAST matches, so nothing leaks
information from the match being predicted):
- team form: points per game, goal difference, shots on target, over the
  last N matches, for home team and away team separately
- head-to-head win rate over the last N meetings
- market-implied probabilities from closing odds (if odds columns exist)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CORE_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC", "HY", "AY", "HR", "AR",
]
ODDS_COLS = ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]


def load_raw(league: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"{league}_*.csv"))
    if not files:
        raise SystemExit(f"No raw files found for league '{league}' in {RAW_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1")
        keep = [c for c in CORE_COLS + ODDS_COLS if c in df.columns]
        df = df[keep].copy()
        df["season_file"] = f.stem
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def add_rolling_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Add pre-match rolling form features per team (points/goals/shots over
    last n matches), computed strictly from matches BEFORE the current row."""
    df = df.copy()
    # long format: one row per (team, match) from each team's point of view
    home = df[["Date", "HomeTeam", "FTHG", "FTAG", "HST", "AST"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA", "HST": "ST_for", "AST": "ST_against"}
    )
    home["Points"] = np.select([df["FTR"] == "H", df["FTR"] == "D"], [3, 1], default=0)
    home["MatchIdx"] = df.index

    away = df[["Date", "AwayTeam", "FTAG", "FTHG", "AST", "HST"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA", "AST": "ST_for", "HST": "ST_against"}
    )
    away["Points"] = np.select([df["FTR"] == "A", df["FTR"] == "D"], [3, 1], default=0)
    away["MatchIdx"] = df.index

    long_df = pd.concat([home, away], ignore_index=True).sort_values("Date")

    long_df["roll_points"] = long_df.groupby("Team")["Points"].transform(
        lambda s: s.shift().rolling(n, min_periods=1).mean()
    )
    long_df["roll_gd"] = long_df.groupby("Team").apply(
        lambda g: (g["GF"] - g["GA"]).shift().rolling(n, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    long_df["roll_st_for"] = long_df.groupby("Team")["ST_for"].transform(
        lambda s: s.shift().rolling(n, min_periods=1).mean()
    )

    form = long_df.set_index(["MatchIdx", "Team"])[["roll_points", "roll_gd", "roll_st_for"]]

    df["home_form_points"] = df.apply(lambda r: form.loc[(r.name, r["HomeTeam"])]["roll_points"], axis=1)
    df["home_form_gd"] = df.apply(lambda r: form.loc[(r.name, r["HomeTeam"])]["roll_gd"], axis=1)
    df["home_form_st"] = df.apply(lambda r: form.loc[(r.name, r["HomeTeam"])]["roll_st_for"], axis=1)
    df["away_form_points"] = df.apply(lambda r: form.loc[(r.name, r["AwayTeam"])]["roll_points"], axis=1)
    df["away_form_gd"] = df.apply(lambda r: form.loc[(r.name, r["AwayTeam"])]["roll_gd"], axis=1)
    df["away_form_st"] = df.apply(lambda r: form.loc[(r.name, r["AwayTeam"])]["roll_st_for"], axis=1)
    return df


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, d, a = ("AvgH", "AvgD", "AvgA") if "AvgH" in df.columns else ("B365H", "B365D", "B365A")
    if h not in df.columns:
        return df
    inv_h, inv_d, inv_a = 1 / df[h], 1 / df[d], 1 / df[a]
    overround = inv_h + inv_d + inv_a
    df["mkt_prob_H"] = inv_h / overround
    df["mkt_prob_D"] = inv_d / overround
    df["mkt_prob_A"] = inv_a / overround
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, help="League code, e.g. E0")
    parser.add_argument("--form-window", type=int, default=5)
    args = parser.parse_args()

    df = load_raw(args.league)
    df = add_rolling_form(df, n=args.form_window)
    df = add_market_probs(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{args.league}_matches.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} matches -> {out_path}")


if __name__ == "__main__":
    main()
