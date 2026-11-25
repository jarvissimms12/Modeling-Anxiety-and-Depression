"""
validate_signal.py — Is there any real signal in this dataset?

Checks every claim a model could make before anyone writes it in a report:
  1. Cross-validated AUC of the Gradient Boosting model (not training accuracy)
  2. Permutation test: how often does a model trained on SHUFFLED labels do as well?
  3. Group comparisons for the interventions (therapy, meditation, substance use)
     with p-values

Run:  python validate_signal.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score

DATA = "anxiety_depression_data.csv"
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA)
    df["Substance_Use"] = df["Substance_Use"].fillna("None")
    df["Medication_Use"] = df["Medication_Use"].fillna("None")

    y = (df["Anxiety_Score"] > df["Anxiety_Score"].median()).astype(int)
    X = pd.get_dummies(df.drop(columns=["Anxiety_Score"]), drop_first=True)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    score, perm_scores, p = permutation_test_score(
        model, X, y, cv=cv, scoring="roc_auc", n_permutations=100,
        random_state=SEED, n_jobs=-1,
    )

    print("=" * 60)
    print("1. Model performance (5-fold cross-validated)")
    print(f"   ROC AUC on real labels:      {score:.3f}")
    print(f"   ROC AUC on shuffled labels:  {perm_scores.mean():.3f} "
          f"(95th pct {np.percentile(perm_scores, 95):.3f})")
    print(f"   Permutation p-value:         {p:.3f}")
    print("   -> AUC 0.5 = coin flip. A p-value above 0.05 means the model")
    print("      does no better than one trained on random labels.")

    print("\n2. Anxiety score by intervention")
    for col in ["Therapy", "Meditation"]:
        a = df.loc[df[col] == 1, "Anxiety_Score"]
        b = df.loc[df[col] == 0, "Anxiety_Score"]
        t = stats.ttest_ind(a, b, equal_var=False)
        print(f"   {col:<11} yes={a.mean():5.2f}  no={b.mean():5.2f}  "
              f"Welch t-test p={t.pvalue:.3f}")
    groups = [g["Anxiety_Score"] for _, g in df.groupby("Substance_Use")]
    means = df.groupby("Substance_Use")["Anxiety_Score"].mean().round(2).to_dict()
    print(f"   Substance   {means}  ANOVA p={stats.f_oneway(*groups).pvalue:.3f}")

    print("\n3. Strongest linear correlations with Anxiety_Score")
    corr = df.select_dtypes("number").corr()["Anxiety_Score"].drop("Anxiety_Score")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(5)
          .round(3).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
