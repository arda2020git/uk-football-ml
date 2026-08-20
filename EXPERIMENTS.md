# Experiment Plan

Each experiment gets its own subfolder in `results/` (e.g. `results/01_baselines/`)
containing: the script/notebook used, a `metrics.json` or `.csv`, and any plots.
Work through them roughly in order — each one introduces one new ML idea on
top of the last, using the same dataset from `data/processed/{league}_matches.csv`.

Target variable throughout: `FTR` (H/D/A), unless noted otherwise.

---

### 01 — Baselines (no learning yet)
**Goal:** know what "good" looks like before you fit anything.
- Predict "Home win" every time. Report accuracy.
- Predict the most frequent class in the training set every time.
- Convert bookmaker closing odds (`mkt_prob_H/D/A` from `build_dataset.py`) into
  a prediction (argmax) and report its accuracy + log-loss. This is the number
  to beat — bookmakers are very hard to beat, and seeing that early sets
  realistic expectations.
- **Concepts:** accuracy, log-loss, class balance, why baselines matter.

### 02 — First classifier: logistic regression
**Goal:** train your first real model.
- Features: home advantage (constant), `home_form_points`, `away_form_points`,
  `home_form_gd`, `away_form_gd`.
- Simple random train/test split (e.g. 80/20) to start — you'll fix this in
  experiment 05.
- **Concepts:** train/test split, feature scaling (`StandardScaler`),
  multiclass logistic regression, confusion matrix.

### 03 — Feature engineering
**Goal:** see how much features matter vs the model itself.
- Add shots on target form, corners, cards, head-to-head history.
- Try encoding teams directly (one-hot or target encoding) vs form-only
  features. Compare accuracy/log-loss to experiment 02.
- **Concepts:** feature engineering, one-hot encoding, overfitting from
  too many sparse features (many teams = many columns).

### 04 — Model comparison
**Goal:** compare a few standard classifiers on the same features.
- Logistic Regression vs Random Forest vs Gradient Boosting
  (`sklearn.ensemble.GradientBoostingClassifier` or `HistGradientBoostingClassifier`)
  vs a simple KNN.
- Use `GridSearchCV`/`RandomizedSearchCV` for basic hyperparameter tuning on
  one of them.
- **Concepts:** model comparison, cross-validation, hyperparameter search,
  bias/variance intuition (why RF overfits small football datasets easily).

### 05 — Doing time series properly
**Goal:** fix the methodology bug from experiments 02-04.
- Football matches are time-ordered — a random split lets the model "see the
  future" (e.g. training on a match from March while testing on one from
  January). Redo experiment 04 with:
  - a chronological train/test split (train on seasons 1..N-1, test on season N), and
  - walk-forward validation (`TimeSeriesSplit` or manual rolling-origin loop).
- Compare metrics to the random-split versions. Expect them to look worse —
  that's the point.
- **Concepts:** data leakage, temporal validation, why this matters more
  than the choice of model.

### 06 — Probability calibration
**Goal:** move from "is the prediction right" to "are the probabilities honest."
- Plot a calibration curve (`sklearn.calibration.calibration_curve`) for your
  best model vs the bookmaker-implied probabilities.
- Try `CalibratedClassifierCV` and see if it helps.
- Report Brier score alongside accuracy/log-loss.
- **Concepts:** why accuracy alone is misleading for probabilistic
  predictions, calibration, Brier score.

### 07 — Handling draws / alternative targets
**Goal:** explore class imbalance and simpler binary targets.
- Draws (~25% of matches) are the hardest class and often ignored by naive
  models. Try `class_weight="balanced"`, or reframe as easier binary problems:
  - "Home win vs not" 
  - "Over/Under 2.5 total goals" (`FTHG + FTAG`)
  - "Both teams to score" 
- Compare how much easier these binary markets are to predict than full 1X2.
- **Concepts:** class imbalance, precision/recall, choosing the right
  problem framing.

### 08 — Feature importance & interpretation
**Goal:** understand *why* your best model predicts what it predicts.
- Permutation importance (`sklearn.inspection.permutation_importance`) on
  your experiment 05 model.
- Optional: SHAP values if you want to go further.
- Sanity-check: does the model mostly rediscover "the market is a strong
  signal" if you include odds as a feature? Try with and without odds
  features to see how much of your model's skill is just "reading the odds."
- **Concepts:** interpretability, sanity-checking a model against domain
  knowledge.

### 09 — A goals-based alternative: Poisson regression
**Goal:** compare a classic domain-specific approach to generic classifiers.
- Fit independent Poisson regressions for home goals and away goals
  (`statsmodels.genmod` or `sklearn` with a Poisson loss) using team
  attack/defence strength as features.
- Derive P(H)/P(D)/P(A) from the two Poisson distributions and compare to
  your experiment 05/06 classifier.
- **Concepts:** generative vs discriminative modelling, Poisson regression,
  a taste of how real sports-analytics models (Dixon-Coles etc.) work.

### 10 — Backtest a simple staking strategy
**Goal:** connect ML metrics to a business metric — and see how different
they are.
- Using your **walk-forward** predictions from experiment 05/06 only (never
  the random-split ones — that would be leaking future info into "profit"),
  simulate flat staking (1 unit per bet) whenever your model's probability
  exceeds the market's implied probability by some margin.
- Report ROI and plot cumulative profit over time. Also compute what ROI
  a random betting strategy would get, as a sanity baseline.
- **Concepts:** expected value, the gap between "accurate" and "profitable,"
  overfitting a backtest, why this is genuinely difficult (markets are
  close to efficient) — treat this step as a lesson in ML methodology and
  healthy skepticism, not betting advice.

---

## Suggested first run

```
python scripts/download_data.py --league E0 --seasons 1516-2425
python scripts/build_dataset.py --league E0
```

Then start a notebook at `notebooks/01_baselines.ipynb` and work through the
list above, saving each experiment's outputs into `results/0N_name/`.
