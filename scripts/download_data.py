"""
Download historical match CSVs from football-data.co.uk.

Examples
--------
# Premier League, 2015-16 through 2024-25
python download_data.py --league E0 --seasons 1516-2425

# Multiple leagues, a single season
python download_data.py --league E0 E1 D1 --seasons 2324
"""
import argparse
import time
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUE_CODES = {
    "E0": "England - Premier League",
    "E1": "England - Championship",
    "E2": "England - League One",
    "E3": "England - League Two",
    "EC": "England - Conference",
    "SC0": "Scotland - Premiership",
    "D1": "Germany - Bundesliga",
    "D2": "Germany - Bundesliga 2",
    "SP1": "Spain - La Liga",
    "SP2": "Spain - La Liga 2",
    "I1": "Italy - Serie A",
    "I2": "Italy - Serie B",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Netherlands - Eredivisie",
    "P1": "Portugal - Primeira Liga",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def season_codes(season_range: str) -> list[str]:
    """Turn '1516-2425' into ['1516', '1617', ..., '2425']."""
    if "-" not in season_range:
        return [season_range]
    start, end = season_range.split("-")
    start_yy = int(start[:2])
    end_yy = int(end[:2])
    return [f"{yy:02d}{yy + 1:02d}" for yy in range(start_yy, end_yy + 1)]


def download(league: str, season: str, dest_dir: Path) -> bool:
    url = BASE_URL.format(season=season, league=league)
    dest = dest_dir / f"{league}_{season}.csv"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200 or not resp.content.strip():
        print(f"  [skip] {url} -> HTTP {resp.status_code}")
        return False
    dest.write_bytes(resp.content)
    print(f"  [ok]   {url} -> {dest}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", nargs="+", required=True, help=f"League code(s). Known: {', '.join(LEAGUE_CODES)}")
    parser.add_argument("--seasons", required=True, help="Single season '2324' or range '1516-2425'")
    parser.add_argument("--out", default=str(DATA_DIR), help="Output directory (default: data/raw)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = season_codes(args.seasons)

    for league in args.league:
        print(f"League {league} ({LEAGUE_CODES.get(league, 'unknown code')}):")
        for season in seasons:
            download(league, season, out_dir)
            time.sleep(0.5)  # be polite to the server


if __name__ == "__main__":
    main()
