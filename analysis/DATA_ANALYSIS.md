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
