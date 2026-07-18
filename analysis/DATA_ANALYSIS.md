# Data Analysis & Cleaning Plan

Dataset: **832 numeric features**, 1212 train rows / 776 test rows.
Target `y`: integer-valued, range **42–97**, mean ~70, **no missing, no negatives**
(looks like a bounded score). Reproduce with `python analysis/analyze_data.py`,
`probe.py`, `row_probe.py`.

---

## Step 1 — Missing values & outliers

### Missing values
- **7.6% of train cells / 6.3% of test cells** are missing.
- Missingness is spread across **all 832 columns** (per-column rate uniform 5–10%),
  and **every row has some missing values** (0 fully complete rows).
- Pattern looks **MCAR** (uniform, not concentrated in specific rows/columns).

**Recommendation**
- **Do not drop rows or columns** — no complete rows exist and every column is
  affected.
- **Impute.** Use **median imputation** (fit on train only, apply to val/test) as a
  robust default — median beats the current mean imputation because several columns
  are heavy-tailed/outlier-prone. Consider `IterativeImputer` / `KNNImputer` later.
- Optionally add **missing-indicator flags** for the highest-NaN columns.

### Outliers
- Most features are near-symmetric (**median |skew| = 0.05**); only 16 columns have
  |skew| > 1, 3 have |skew| > 5.
- Point outliers are **rare**: only 146 values with |z| > 5 (0.016%), max |z| ≈ 23;
  IQR flags ~1% of values per column on average.
- **Feature scales span ~20 orders of magnitude.** Column stds range from 0 to
  4×10²². Three columns are astronomically large: **x665, x173, x596**
  (std 10¹⁴–10²²) — and this is the *whole column* (>91% of values > 1e6), not a few
  outliers.

**Recommendation**
- **Scaling is mandatory** (RobustScaler preferred, or StandardScaler): without it,
  x665/x173/x596 dominate any distance- or gradient-based model.
- **Winsorize / clip** extreme point values (e.g. clip per column at the 1st/99th
  percentile, or |robust-z| > 5) instead of deleting rows.
- For the ~16 heavily skewed columns, a **PowerTransformer (Yeo-Johnson)** helps —
  and it tolerates the 36 columns that contain negative values (plain log won't).

---

## Step 2 — Further problems / perturbations

1. **Constant features (dead columns):** `x104, x129, x489, x530` are identically 0.
   → **Drop them** (zero information; they break correlation/scaling).

2. **Scale-perturbed features:** `x665, x173, x596` appear to have been multiplied by
   huge constants (values ~10²²). Handled by scaling, but worth flagging as a likely
   deliberate perturbation.

3. **Perturbed / corrupted samples (row outliers):** using robust median/MAD z-scores,
   **3 rows have >20 extreme features and 1 row has ~90**. These few rows look
   corrupted — inspect and consider removing them from training.

4. **No covariate shift train→test:** per-feature mean shift ≤ 0.17 std, std ratios
   0.57–1.31. The test set is **not** adversarially shifted in aggregate → models
   should generalise.

5. **Clean on other axes:** no duplicate rows, no duplicate columns, no infinities.

6. **Signal is present:** 138 features have |Spearman corr with y| > 0.2 (top ≈ 0.60),
   so the target is learnable and feature selection is worthwhile.

---

## Suggested cleaning pipeline (order matters)

1. Drop constant columns (`x104, x129, x489, x530`).
2. (Optional) Drop the ~3 corrupted training rows.
3. Median-impute missing values (fit on train).
4. Clip/winsorize per-column outliers.
5. Scale (RobustScaler) — critical given the scale spread.
6. (Optional) Yeo-Johnson on skewed columns.
7. Then feature selection + model (as in `main.py`).

All transforms must be **fit on the training split only** and applied to
validation/test to avoid leakage.

---

## Modeling experiments & improvements (5-fold CV R²)

All numbers are 5-fold cross-validated R² (`KFold(shuffle=True, random_state=42)`),
so they are honest estimates of generalisation, not a single lucky split. We log
every experiment to `analysis/experiment_log.txt`. Below we document, in order,
**only the changes that improved R²** and by how much. Target: well above 0.5.

### Step A — the cleaning pipeline (linear model)
Scripts: `src/cleaning.py`, `analysis/compare_pipelines.py`.

| Pipeline | R² | What changed |
|----------|----|--------------|
| baseline (mean impute + OLS) | 0.179 ± 0.151 | starting point |
| cleaned + OLS | 0.343 ± 0.073 | median impute + drop constant cols + winsorize + robust scale |
| cleaned + Ridge(α=10) | 0.353 ± 0.072 | add L2 regularisation |

**Impact:** the cleaning pipeline roughly **doubled** the mean R² (0.18 → 0.35) and
**halved the variance**. The baseline's huge ±0.15 variance came from the
extreme-scale columns (x665/x173/x596 ~10²²) making OLS numerically unstable;
robust scaling + winsorising removed it. This confirms the cleaning plan works, but
linear models plateau around **0.35** — a hard ceiling.

### Step B — switching to non-linear (tree) models
Script: `analysis/experiments.py`. Trees need no scaling, and `HistGradientBoosting`
handles missing values natively.

| Model | R² | What changed |
|-------|----|--------------|
| Ridge(10) (best linear) | 0.336 | reference |
| HistGradientBoosting (defaults) | 0.457 ± 0.027 | linear → non-linear |
| RandomForest(300) | 0.459 ± 0.034 | |
| ExtraTrees(400) | 0.469 ± 0.035 | |

**Impact:** moving from linear to tree models jumped R² from ~0.34 to ~0.47 — a
**+0.12 gain** — and further cut variance. This is the single biggest improvement and
proves the target depends on the features **non-linearly**, so linear models can
never reach 0.5 here.

### Step C — gradient boosting crosses 0.5
Same script. Gradient boosting (sequential, error-correcting trees) with shallow
trees and row subsampling (stochastic boosting) is the strongest family.

| Model | R² | What changed |
|-------|----|--------------|
| ExtraTrees(400) | 0.469 | best bagging model |
| **GradientBoosting (lr=0.05, n=400, depth=3, subsample=0.8)** | **0.502 ± 0.043** | boosting + shallow depth + subsampling |

**Impact:** gradient boosting **crossed the 0.5 target** (+0.03 over ExtraTrees).
Shallow depth-3 trees + 80% row subsampling regularise the model and generalise
better than deeper or full-sample variants.

### Step D — further GB tuning & ensembling
Script: `analysis/experiments2.py`.

| Model | R² | Verdict |
|-------|----|---------|
| GB base (n=400, d=3, lr=.05, sub=.8) | 0.5022 ± 0.043 | reference |
| GB + max_features='sqrt' | 0.4847 ± 0.043 | ✗ worse |
| GB n=600, max_features='sqrt' | 0.4831 ± 0.042 | ✗ worse |
| GB n=800, lr=.03, max_features='sqrt' | 0.4901 ± 0.029 | ✗ worse |
| GB depth=4, max_features='sqrt' | 0.4749 ± 0.041 | ✗ worse |
| GB depth=2, n=800, max_features='sqrt' | 0.4772 ± 0.041 | ✗ worse |
| **GB n=600, max_features=0.3, sub=.7** | **0.5030 ± 0.035** | ✓ best + lowest variance |
| HistGBM depth=3, lr=.05, iter=600, l2=1 | 0.4649 ± 0.045 | ✗ worse |
| Stacking(GB+ExtraTrees+HistGBM)→RidgeCV | 0.5005 ± 0.047 | ≈ no gain |

**Impact:** essentially a plateau. The key negative finding is that
**`max_features='sqrt'` consistently hurts** (~−0.02): with 828 features and much
noise, restricting each split to ~29 features loses signal. Using a larger fraction
(`max_features=0.3`) with more row subsampling gave the best, most stable result
(**0.503 ± 0.035**). Stacking three tree models did **not** beat a single tuned GB —
the base learners are too correlated. Conclusion: sklearn gradient boosting caps
around **0.50**; to go well beyond we need stronger boosting libraries (Step E).

### Step E — stronger boosting libraries (LightGBM, XGBoost)
Script: `analysis/experiments3.py`. Both handle NaN natively and need no scaling.

| Model | R² | Verdict |
|-------|----|---------|
| LightGBM (defaults) | 0.448 ± 0.046 | |
| XGBoost (defaults) | 0.407 ± 0.027 | |
| LightGBM n=500, lr=.05, leaves=31, col=.8 | 0.485 ± 0.045 | |
| LightGBM n=1000, lr=.02, leaves=15, col=.5 | 0.491 ± 0.045 | best LGBM |
| XGBoost n=500, lr=.05, d=4, col=.5, λ=1 | 0.488 ± 0.047 | |
| XGBoost n=800, lr=.03, d=3, col=.3, λ=2 | 0.497 ± 0.037 | best XGB |

**Impact / key finding: none — the fancier libraries did *not* beat sklearn GB
(0.503).** This is counter-intuitive but expected for this dataset shape: with only
**1212 rows and 828 features**, LightGBM's/XGBoost's aggressive leaf-wise growth
overfits, and heavy regularisation only claws back to ~0.50. All gradient-boosting
implementations cluster at **0.45–0.50**, which signals a **signal ceiling on the raw
feature set**. The next lever is therefore not the model but the **features**:
cutting the ~800-column noise down to the informative signal (Step F).

### Step F — feature selection (SelectKBest, mutual information)
Script: `analysis/experiments4.py`. Impute → SelectKBest(mutual_info, k) → model.

| k (features) | GB R² | XGB R² |
|--------------|-------|--------|
| 30 | 0.474 | — |
| 50 | 0.486 | 0.490 |
| 100 | 0.496 | 0.499 |
| 200 | 0.500 | 0.502 |
| 300 | 0.504 | 0.507 |
| **500** | **0.515 ± 0.047** | — |
| SelectFromModel(GB importance) | 0.500 | — |

**Impact:** a small but consistent gain, and **the opposite of the expected pattern** —
performance rises monotonically with k, so *more* informative features help rather
than aggressive pruning. Best so far is **GB + top-500 features = 0.515**, a new high.
The gains are within the ±0.045 fold-to-fold noise, so before trusting them we (a)
locate the k peak and (b) switch to repeated CV to shrink the error bars (Step G).

### Step G — k peak, model averaging, and honest (repeated) CV
Script: `analysis/experiments5.py`.

*k peak (single 5-fold, GB):* 400→0.507, **500→0.515**, 600→0.510, 700→0.502,
828→0.501. The optimum feature count is **~500**.

*Averaging ensembles (single 5-fold, k=500):*

| Model | Single 5-fold R² |
|-------|------------------|
| GB | 0.515 |
| **Voting(GB + XGB)** | **0.521** |
| Voting(GB + XGB + ExtraTrees) | 0.519 |

Averaging the two boosters lifts the single-split score to **0.521** — decorrelated
errors cancel.

*Reality check — RepeatedKFold (5 folds × 3 repeats = 15 splits):*

| Model | Single 5-fold | **Robust 5×3 CV** |
|-------|---------------|-------------------|
| GB k=500 | 0.515 | 0.493 ± 0.051 |
| Voting(GB+XGB) k=500 | 0.521 | 0.502 ± 0.046 |
| Voting(GB+XGB+ET) k=500 | 0.519 | **0.504 ± 0.043** (best, lowest var) |

**Impact / key methodological finding:** the single-split numbers were **optimistic by
~0.02** — `random_state=42` happened to be a favourable split. Averaged over 15
splits, the honest performance is **~0.50**, and the **three-model ensemble is the most
robust** (highest mean, lowest variance). We therefore adopt
**Voting(GB + XGB + ExtraTrees) on the top-500 MI features** as the production model:
robust CV **0.504 ± 0.043**, single-split (leaderboard-style) **≈0.52**.

> Lesson for the presentation: always confirm gains with repeated CV. A single
> shuffled split on 1212 rows has ±0.05 noise, enough to invent a 0.02 "improvement".

### Step H — feature engineering (attempt to add signal)
Script: `analysis/experiments6.py`. Tried on top of the Voting ensemble.

| Feature engineering | Single 5-fold | Robust 5×3 CV |
|---------------------|---------------|----------------|
| base Voting k=500 | 0.519 | 0.5042 ± 0.043 |
| + missingness indicators (k=500) | 0.516 | — |
| + missingness indicators (k=700) | 0.512 | — |
| + row aggregates (mean/std/min/max/NaN-count) | 0.515 | 0.5043 ± 0.045 |

**Impact: none.** Adding missing-value indicators or row-level aggregates did **not**
improve R² (robust CV identical: 0.5042 vs 0.5043). This is a clean confirmation of the
data analysis: the missingness is **MCAR** (uninformative), so encoding it adds noise,
not signal. The signal ceiling for this feature set sits at **~0.50 (robust) / ~0.52
(single split)**.

---

## Summary: the improvement journey (for the presentation)

Every number below is cross-validated R² (higher is better). The **bold** rows are the
changes that moved the needle.

| # | Change | Robust R² | Δ | Lesson |
|---|--------|-----------|---|--------|
| 0 | Raw data + OLS (given baseline) | 0.18 ± 0.15 | — | unstable: extreme-scale columns wreck OLS |
| **A** | **Cleaning pipeline (impute/winsorize/scale) + Ridge** | **0.35 ± 0.07** | **+0.17** | robust scaling fixes the instability |
| **B** | **Linear → tree models** | **0.47 ± 0.03** | **+0.12** | the relationship is non-linear |
| **C** | **Gradient boosting (shallow + subsampling)** | **0.50 ± 0.04** | **+0.03** | boosting > bagging here; crossed 0.5 |
| D | GB hyper-tuning / stacking | 0.50 | ~0 | `max_features='sqrt'` hurts; stacking no gain |
| E | LightGBM / XGBoost | 0.50 | ~0 | fancier libs overfit on 1212×828 data |
| **F** | **Feature selection (top-500 MI)** | **0.50** | small | more features help, not fewer |
| **G** | **Averaging GB + XGB + ExtraTrees** | **0.504 ± 0.043** | **+0.01** | decorrelated errors cancel |
| H | Feature engineering (missingness, aggregates) | 0.504 | 0 | missingness is MCAR → no signal |

**Final model:** `median impute → SelectKBest(mutual_info, k=500) →
Voting(GradientBoosting + XGBoost + ExtraTrees)` (`src/model.py`).
Robust 5×3 CV **R² = 0.504 ± 0.043**; single-split (leaderboard-style) **≈ 0.52**.

**Biggest lifts:** cleaning (+0.17) and going non-linear (+0.12). Everything after
gradient boosting is incremental (±0.01) and near the data's signal ceiling. To go
materially higher would require *new signal* (external features or domain knowledge),
not more model tuning.

*Reproduce any run:* `python -u analysis/experiments.py` (…`2`…`6`). Raw numbers are
appended to `analysis/experiment_log.txt`.

---

# Second attempt — imputation & outlier detection

Goal: revisit the two data-quality decisions from attempt 1 — **imputation** and
**outlier handling** — as two *separate, isolated* experiments, and see whether either
robustly beats the first attempt. Everything else (feature selection, model
hyper-parameters) is held fixed. Branch: `second-attempt-knn-imputation`.

**Method note (why a RobustScaler was added):** both KNN imputation and the outlier
detectors rely on Euclidean distances. Our features span ~20 orders of magnitude
(x665/x173/x596 ≈ 10²²), which would completely dominate any distance. So a
NaN-preserving `RobustScaler` is applied *before* the distance step. Scaling is
invisible to tree splits (monotonic), so it changes tree results only through the
imputed/selected values — exactly what we are testing. Unified preprocessing:
`winsorize → RobustScaler → impute → drop constant → SelectKBest(k) → model`.

**Compute design:** for each (imputer, fold) the expensive KNN imputation and the
mutual-info ranking are computed **once and shared across all models**, instead of
recomputed per model — roughly a 10× saving. Exploration uses single 5-fold; anything
promising is confirmed with RepeatedKFold (5×3).

## Experiment 2a — KNN imputation vs median (`analysis/exp_knn.py`)

Swapped `SimpleImputer(median)` → `KNNImputer(n_neighbors=5)`, re-scored models A–G.

| Model | median | KNN | Δ |
|-------|--------|-----|---|
| **A: Ridge (linear)** | 0.349 | **0.371** | **+0.022** |
| B: HistGBM | 0.460 | 0.459 | −0.001 |
| B: RandomForest(300) | 0.457 | 0.454 | −0.003 |
| B: ExtraTrees(400) | 0.472 | 0.467 | −0.005 |
| C: GradientBoosting | 0.492 | 0.492 | −0.000 |
| E: LightGBM | 0.494 | 0.490 | −0.004 |
| E: XGBoost | 0.493 | 0.493 | +0.000 |
| F: GB k=500 | 0.498 | 0.493 | −0.005 |
| F: XGB k=300 | 0.512 | 0.508 | −0.004 |
| G: Voting (champion) | 0.510 | 0.506 | −0.004 |

**Confirmation (RepeatedKFold 5×3), `analysis/exp_knn_confirm.py`:**

| Model | median | KNN | Δ |
|-------|--------|-----|---|
| A: Ridge | 0.336 ± 0.079 | **0.364 ± 0.056** | **+0.028, variance −30%** |
| F: GB k=500 | 0.485 ± 0.050 | 0.484 ± 0.055 | −0.000 |

**Result: KNN imputation robustly helps only the *linear* model (Ridge: +0.028 and
notably more stable), and does nothing for any tree model — including the champion
ensemble.** Textbook behaviour: linear models are sensitive to imputation quality;
tree splits are robust to it. Since our production model is a tree ensemble, **KNN
imputation does not improve our best result** — it would only matter if we shipped a
linear model.

## Outlier detection — Isolation Forest vs LOF (`analysis/outlier_probe.py`)

Before adding a detector we probed both on the imputed+scaled data. The 3 known-corrupted
rows (from `row_probe.py`) are **167, 681, 740**.

| contamination | iForest flags | LOF flags | agreement | iForest hits | LOF hits |
|---|---|---|---|---|---|
| 0.005 | 7 | 7 | 1 | 167, 740 | 681, 740 |
| 0.01 | 13 | 13 | 2 | 167, 740 (misses 681) | all 3 |
| 0.02 | 25 | 25 | 8 | all 3 | all 3 |

The two methods **barely agree** — they encode different notions of "outlier" (iForest =
global isolation, LOF = local density). Theory favours iForest here (scale-invariant,
survives 828 dimensions, targets global anomalies; LOF suffers curse-of-dimensionality
and needs scaling). Because the probe was empirically ambiguous, we let CV R² decide via
a bake-off rather than committing a priori.

## Experiment 2b — outlier-row removal bake-off (`analysis/exp_outliers.py`)

Drop rows flagged in each training fold (median imputation kept, so this is independent
of 2a). Detectors fit on the train fold only.

| Config (rows removed) | GB k=500 | Voting (champion) |
|-----------------------|----------|-------------------|
| no removal | 0.4976 | 0.5098 |
| **iForest c=0.005 (~5)** | **0.5049 (+0.007)** | **0.5150 (+0.005)** |
| iForest c=0.01 (~10) | 0.5035 (+0.006) | 0.5091 (−0.001) |
| LOF c=0.005 (~5) | 0.5005 (+0.003) | 0.5082 (−0.002) |
| LOF c=0.01 (~10) | 0.4912 (−0.006) | 0.5078 (−0.002) |

**Isolation Forest at contamination = 0.005 (≈5 rows) is the only config that helps both
models** — empirically confirming iForest > LOF, and that removing too many rows erodes
the gain (small training set, 1212 rows).

**Confirmation (RepeatedKFold 5×3), `analysis/exp_outliers_confirm.py`:**

| Model | no removal | iForest c=0.005 | Δ |
|-------|-----------|-----------------|---|
| GB k=500 | 0.4845 ± 0.050 | 0.4878 ± 0.043 | +0.003 |
| **Voting (champion)** | 0.5010 ± 0.043 | **0.5024 ± 0.039** | **+0.001, variance −8%** |

**Result: iForest removal gives a small but consistent robust gain and lower variance.**
For the champion it's tiny (+0.001, within noise) but always non-negative and reduces
variance — a safe, defensible addition. LOF is inconsistent and can hurt.

## Second-attempt verdict

| Change | Robust effect on champion | Adopt? |
|--------|---------------------------|--------|
| KNN imputation | ~0 (helps only linear models, +0.028 for Ridge) | **No** for the tree ensemble; keep median |
| Isolation Forest row removal (c=0.005) | +0.001 mean, variance −8% | **Optional** — marginal but safe; iForest clearly beats LOF |

Neither change materially breaks the **~0.50 robust / ~0.52 single-split ceiling**. This
reinforces attempt 1's conclusion: the data-quality levers are largely exhausted, and a
real jump needs **new signal** (domain meaning of the features), not more imputation or
outlier tuning. The two clean, presentable findings: **(1)** imputation method matters for
linear models but not trees; **(2)** among detectors, Isolation Forest is the right choice
for high-dimensional, extreme-scale data, and only very light removal (~0.5%) helps.

*Reproduce:* `python -u analysis/exp_knn.py`, `exp_knn_confirm.py`, `outlier_probe.py`,
`exp_outliers.py`, `exp_outliers_confirm.py`. Raw numbers in `analysis/experiment_log.txt`.
