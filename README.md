# Poisoning-Loop-with-Fairness-Monitoring

# Part 4: Stress Testing the COMPAS Model

Extension of the COMPAS fairness audit (Part 3). Covers distribution drift, generalization, spurious correlation detection, robustness testing, and slice-based evaluation using the same logistic regression model and dataset from earlier parts.

---

## Contents

| File | Description |
|------|-------------|
| `Part4_Clean.ipynb` | Main notebook — all code, outputs, and interpretations |

---

## How to Run

1. Open `Code_Part3_Disparate_Impact.ipynb` in Google Colab and run all cells
2. Paste the cells from `Part4_Clean.ipynb` at the bottom of that file
3. Run the Setup cell first, then Parts A through E in order

All Part 4 cells depend on variables created in Part 3. Do not run Part 4 in isolation.

---

## Parts

### Part A: Distribution Drift

Checks whether the feature distributions in the training split match the test split. Three metrics are used:

- **PSI** (Population Stability Index) — tests each numeric feature independently. Thresholds: below 0.10 stable, 0.10–0.25 slight shift, above 0.25 major shift.
- **KS test** — two-sample Kolmogorov-Smirnov test for distributional equality. p < 0.05 indicates significant drift.
- **MMD** (Maximum Mean Discrepancy) — measures distance between full joint distributions in an encoded feature space using an RBF kernel. Catches correlated feature drift that PSI and KS miss.

Also compares the distribution of predicted scores (model output) between train and test.

### Part B: Generalization

Compares model performance on training data versus held-out test data to diagnose overfitting.

Metrics compared: AUC, accuracy, and log loss. Gap thresholds: AUC gap above 0.02 or log loss gap above 0.05 flags overfitting. Includes ROC curves for both splits and a calibration plot showing whether predicted probabilities match actual outcome rates on the test set.

### Part C: Spurious Correlation Probe

Tests whether protected attributes carry direct predictive weight in the model. For each attribute, a counterfactual copy of the relevant defendants is created with only that attribute changed, and predictions are compared before and after.

Three swaps tested:
- Race: African-American to Caucasian
- Sex: Male to Female
- Age: Less than 25 to 25–45

Results show mean probability change, standard deviation of change, and the number of defendants whose predicted class flips.

### Part D: Robustness

**Stress test:** replaces every test-set defendant's `priors_count` with values 0 through 30 and tracks mean predicted probability and high-score rate at each level. Reports the threshold crossing point — the priors count at which the average defendant tips into the high-risk category.

**ICE curves** (Individual Conditional Expectation): shows the same priors_count relationship for 60 sampled individual defendants. One panel shows all individuals, the second colours lines by race to show whether baseline predicted probabilities differ by race at the same priors count.

**Sensitivity summary:** reports the linear slope (probability increase per additional prior), R-squared, and max marginal effect.

### Part E: Slice-based Evaluation

Computes performance metrics separately for each racial group, each sex, and each age category. Metrics per slice: AUC, accuracy, FPR, FNR, selection rate, and AIR (Adverse Impact Ratio).

AIR is the group selection rate divided by the reference group selection rate. Below 0.80 triggers the EEOC four-fifths rule. Reference groups: Caucasian for race, Male for sex, 25–45 for age.

Outputs include a full accountability table, a heatmap of all metrics by race, FPR and FNR bar charts by race, and an AUC range check across all groups.

---

## Key Findings

- No distributional drift between train and test — the random stratified split produced balanced samples across all numeric features (Part A)
- The model generalizes well with near-zero AUC and accuracy gaps (Part B)
- Race, sex, and age all carry direct predictive weight confirmed by counterfactual swaps; African-American defendants show the largest mean probability drop when race is switched to Caucasian (Part C)
- `priors_count` is the strongest continuous predictor; its effect varies across defendants depending on other features, visible in the spread of ICE curves (Part D)
- FPR is higher for African-American defendants than for Caucasian defendants; AIR falls below 0.80 for multiple racial groups (Part E)

---

## Dataset

ProPublica COMPAS Analysis dataset — 6,172 defendants after filtering.

Source: https://github.com/propublica/compas-analysis

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
statsmodels
```

---

## Acknowledgment

Generative AI (Claude) was used to assist with code debugging and document writing during this assignment.
