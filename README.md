# 🧠 Modeling Anxiety & Depression — A Null-Result Case Study

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

I set out to find which lifestyle interventions (therapy, meditation, substance use)
predict anxiety scores, using Gradient Boosting with SHAP explainability. **The
answer is that this dataset doesn't support any of those conclusions.** This repo
shows how I checked, so a model's output never goes into a report without
evidence behind it.

---

## Results

`python validate_signal.py` (1,200 respondents, target = anxiety above the median):

| Check | Result | What it means |
|---|---|---|
| Gradient Boosting, 5-fold CV ROC AUC | **0.476** | No better than a coin flip (0.5) |
| Same model on shuffled labels (100 permutations) | 0.497 | Real labels do no better than random ones |
| Permutation test p-value | **0.82** | No evidence of signal |
| Therapy vs. no therapy | 10.60 vs 10.44, p = 0.69 | No difference |
| Meditation vs. none | 10.30 vs 10.58, p = 0.42 | No difference |
| Substance use (none / occasional / frequent) | 10.48 / 10.74 / 9.85, ANOVA p = 0.40 | No difference |
| Strongest correlation with anxiety (any feature) | \|r\| = 0.074 | Effectively zero |

**Conclusion:** the features in this dataset carry no information about anxiety
scores. The variables look independently generated, which is common in synthetic
practice datasets. SHAP values from a model with no predictive power still produce
a tidy-looking ranked chart. That chart describes noise and shouldn't be read
as "therapy matters most."

> An earlier version of this README reported therapy and substance use as the top
> predictors. Those claims came from reading SHAP plots without checking
> out-of-sample performance first. They were wrong, and the checks above are
> what I now run before interpreting any model.

---

## What this project demonstrates

- **Validating before interpreting:** cross-validated AUC and a permutation test come before any feature-importance chart
- **Hypothesis testing:** Welch t-tests and one-way ANOVA for group comparisons
- **Explainable ML:** Gradient Boosting + SHAP (`gradient_boosting_&_shap.py`), and its limits
- **Reporting honestly:** stating a null result clearly, which matters in health research where false positives drive bad decisions

---

## How to run

```bash
git clone https://github.com/jarvissimms12/Modeling-Anxiety-and-Depression.git
cd Modeling-Anxiety-and-Depression
pip install pandas seaborn matplotlib scikit-learn shap scipy

python validate_signal.py              # signal checks (start here)
python "Coping Strategies.py"          # segmentation + visualizations
python gradient_boosting_&_shap.py     # GBM + SHAP plots
```

The dataset (`anxiety_depression_data.csv`) is included.

## Project structure

```
Modeling-Anxiety-and-Depression/
├── validate_signal.py                # CV AUC, permutation test, group tests
├── Coping Strategies.py              # segmentation & visualizations
├── gradient_boosting_&_shap.py       # GBM model + SHAP
├── anxiety_depression_data.csv
└── README.md
```

## Related work

For mental and physical health analysis on **real** public data, see
[state-of-nyc](https://github.com/jarvissimms12/state-of-nyc), which links housing conditions,
schools and city jobs across NYC community districts.

---

**Jarvis Simms** | MS Data Science, NYIT 2025 | Brooklyn, NY ·
[GitHub](https://github.com/jarvissimms12) · [LinkedIn](https://www.linkedin.com/in/jarvis-simms-846062136/)
