# Exerscise 5: Compas aml audit

Adversarial machine learning audit on the ProPublica COMPAS recidivism dataset.
Coursework for DNSC 6330 (Responsible Machine Learning), Homework 5.

The audit covers three attack families on two target models (logistic regression
and gradient-boosted trees):

1. PGD evasion at the deployment layer
2. Label-flip poisoning at the training layer
3. Shadow-model membership inference

Each attack is evaluated for both predictive impact (AUC) and fairness impact
(false-positive rate by race, adverse impact ratio).

## Files

```
.
├── HW5_Exercise5.ipynb   Colab notebook with the full pipeline
├── HW5_Report.docx       Two-page written report
├── README.md             this file
├── LICENSE               MIT
└── .gitignore
```

The notebook is self-contained: it pulls COMPAS directly from the ProPublica
GitHub mirror and trains everything from scratch. No external data files are
required.

## Reproducibility

Open in Google Colab or run locally with Python 3.10+.

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook HW5_Exercise5.ipynb
```

All randomness is seeded with `random_state=42`. Expected runtime on a free
Colab CPU is roughly 4 to 6 minutes; the slowest section is the numerical-
gradient PGD on GBT.

## Methods

Data preprocessing follows the ProPublica conventions: filter to records with
`days_b_screening_arrest` between -30 and 30, drop ordinary traffic charges,
exclude `is_recid == -1`. Features are age, prior counts, juvenile counts, and
one-hot indicators for charge degree and sex. The label is `two_year_recid`.

The same train/test split (70/30, stratified on the target) is used throughout.

### Part 1: PGD evasion

For logistic regression, PGD uses the closed-form gradient sign of the binary
cross-entropy loss. For GBT, PGD uses a numerical gradient computed by central
finite differences on the predicted probability. Epsilons tested:
{0, 0.25, 0.5, 1.0, 2.0}. The attack is untargeted and L-infinity bounded.

### Part 2: Label-flip poisoning

The attacker flips a fraction of training labels for one race, from
`two_year_recid=1` to `two_year_recid=0`. The loop is run twice, once with
`target_race='African-American'` and once with `target_race='Caucasian'`, at
poison rates {0%, 2%, 5%, 8%, 10%, 15%, 20%, 25%, 30%}.

A PSI drift monitor is implemented and applied both to input features and to
output prediction scores, with the conventional alarm threshold of 0.10.

### Part 3: Membership inference

Shokri-style shadow attack: 10 shadow models trained on random 50/50 splits of
the training set, a decision-tree meta-classifier trained on
(max-confidence, member?) pairs, then evaluated on the target model's
train-vs-test outputs. Shadow architecture matches the target.

The L2 sweep refits LR with `C ∈ {0.01, 0.1, 1.0, 10.0}` and reruns the full
shadow pipeline at each setting.

## Headline results

The COMPAS-trained models violate the four-fifths rule before any attack:

| Model | Test AUC | FPR (AA) | FPR (CA) | AIR  |
|-------|---------:|---------:|---------:|-----:|
| LR    | 0.735    | 0.281    | 0.143    | 1.96 |
| GBT   | 0.718    | 0.317    | 0.178    | 1.78 |

Selected findings:

- LR collapses smoothly under PGD. By eps=2.0, every test record is classified
  as recidivist (FPR=1.0 for both groups). AIR converges to 1.0 only because
  the model has stopped distinguishing anyone.
- GBT does not move at all under the implemented numerical-gradient PGD. This
  reflects an attack failure (tree predictions are piecewise-constant in
  feature space and small perturbations rarely cross splits), not robustness.
  A transferability or decision-based attack would be required for a fair
  comparison.
- Targeting African-American defendants for label-flip poisoning at 30% pushes
  AIR from 1.96 to 3.01 while AUC moves only 0.4 percentage points. The
  attack is invisible to AUC monitoring.
- Feature-level PSI cannot detect label-flip poisoning at any rate, because
  the features are unchanged. Output-distribution PSI grows with poison rate
  but only crosses the 0.10 threshold at high rates.
- Membership inference AUC is 0.497 (LR) and 0.500 (GBT), essentially random,
  despite GBT's +0.080 generalization gap. Generalization gap measured in AUC
  does not predict MI vulnerability on this dataset.
- The L2 sweep is a null result: MI AUC, test AUC, and AIR all stay flat
  across two orders of magnitude in C.

The two-page report explains each finding and proposes one proactive
mitigation (AIR-based deployment gate) and one reactive mitigation
(output-distribution PSI monitoring), with quantified effects taken directly
from the experimental tables.

## Limitations

- Numerical-gradient PGD is the wrong attack for non-differentiable models.
  GBT robustness should not be inferred from this experiment.
- The "stealth zone" stealth criterion as defined in the assignment is
  degenerate when the baseline AIR already lies outside [0.80, 1.25]. Results
  are reported in the spirit of the question by tracking the change in AIR
  rather than crossings of the threshold.
- COMPAS is a small dataset (about 6,000 rows after filtering). MI results may
  not generalize to larger datasets where overfitting is more pronounced.
- Mitigations are evaluated only against the attacks tested here. Adaptive
  attackers aware of the monitoring scheme could design poisoning rates and
  attack distributions to evade the proposed PSI threshold.

## References

- Angwin, Larson, Mattu, Kirchner. "Machine Bias." ProPublica, 2016.
  Data: https://github.com/propublica/compas-analysis
- Madry, Makelov, Schmidt, Tsipras, Vladu. "Towards deep learning models
  resistant to adversarial attacks." ICLR 2018.
- Shokri, Stronati, Song, Shmatikov. "Membership inference attacks against
  machine learning models." IEEE S&P 2017.
- Vassilev, Oprea, Fordyce, Anderson. "Adversarial Machine Learning: A
  Taxonomy and Terminology of Attacks and Mitigations." NIST AI 100-2 E2023.

## License

MIT. See `LICENSE`.
