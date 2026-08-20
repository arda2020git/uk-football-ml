# UK Football Prediction — ML Learning Project

Learning the basics of machine learning by predicting football match outcomes
using historical data from [football-data.co.uk](https://www.football-data.co.uk/data.php).

## Folder structure

```
uk_football/
├── data/
│   ├── raw/            # CSVs downloaded as-is from football-data.co.uk
│   └── processed/      # cleaned / merged / feature-engineered datasets
├── scripts/             # reusable Python scripts (download, features, training)
├── notebooks/           # exploratory notebooks, one per experiment if you like
├── results/              # one subfolder per experiment: metrics, plots, saved models
├── requirements.txt
├── environment.yml      # mamba/conda environment spec
└── EXPERIMENTS.md       # the experiment plan (start here)
```

## Environment setup (mamba)

```
mamba env create -f environment.yml
mamba activate uk-football-ml
```

This installs Python 3.11 with pandas, numpy, scikit-learn, matplotlib,
seaborn, statsmodels, requests, and JupyterLab. A Jupyter kernel named
"Python (uk-football-ml)" is registered so notebooks in `notebooks/` can
use this environment.

If you'd rather use plain `pip`/`venv`, `requirements.txt` covers the
same core packages.

## Getting the data

football-data.co.uk publishes one CSV per league per season, e.g.:

```
https://www.football-data.co.uk/mmz4281/2324/E0.csv   # Premier League 2023-24
https://www.football-data.co.uk/mmz4281/2223/E0.csv   # Premier League 2022-23
```

Run:

```
pip install -r requirements.txt
python scripts/download_data.py --league E0 --seasons 1516-2425
```

This saves files into `data/raw/`. See `scripts/download_data.py` for the
full list of league codes (E0 = Premier League, E1 = Championship, D1 =
Bundesliga, SP1 = La Liga, I1 = Serie A, F1 = Ligue 1, ...).

## Column glossary (most useful columns)

- `FTHG`, `FTAG`, `FTR` — full-time home/away goals, result (H/D/A)
- `HTHG`, `HTAG`, `HTR` — half-time equivalents
- `HS`, `AS`, `HST`, `AST` — shots / shots on target
- `HC`, `AC` — corners
- `HY`, `AY`, `HR`, `AR` — cards
- `B365H`, `B365D`, `B365A` — Bet365 closing odds (home/draw/away)
- `AvgH`, `AvgD`, `AvgA` (newer files) — market-average closing odds
- Full glossary: https://www.football-data.co.uk/notes.txt

## Where to start

Read `EXPERIMENTS.md` for a sequenced set of experiments, from a naive
baseline up to a backtested betting strategy. Each experiment gets its own
folder under `results/`.
