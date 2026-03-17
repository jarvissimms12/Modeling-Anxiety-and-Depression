🧠 Modeling Anxiety & Depression
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
A statistical analysis and predictive modeling project investigating the relationship between lifestyle interventions (therapy, meditation, substance use) and anxiety/depression outcomes. Uses Gradient Boosting with SHAP explainability to identify which behavioral factors most influence mental health scores — and segments the population to reveal how high-anxiety and low-anxiety groups differ.
---
🎯 Project Highlights
Feature	Detail
Analysis Type	Segmentation, correlation analysis, predictive modeling
Target Variable	Anxiety score (High vs. Low, split at median)
Models	Gradient Boosting + SHAP explainability
Key Interventions Studied	Therapy, Meditation, Substance Use
Visualization	seaborn, matplotlib (grouped comparisons, feature importance)
Dataset	Included (`anxiety_depression_data.csv`)
---
🔍 Research Questions
Which lifestyle interventions are most associated with lower anxiety scores?
How do High-Anxiety and Low-Anxiety groups differ in their behavioral profiles?
What features does a Gradient Boosting model identify as most predictive?
---
🧠 Technical Approach
Population Segmentation
The dataset is split into High Anxiety and Low Anxiety groups based on the median anxiety score, enabling direct comparison of lifestyle factor distributions across groups.
Gradient Boosting + SHAP
```python
# Model pipeline
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

# Explainability
shap.TreeExplainer(model)  →  SHAP values per feature per prediction
```
SHAP (SHapley Additive exPlanations) translates model outputs into human-readable feature contributions — critical for communicating findings to non-technical stakeholders.
---
📊 Key Findings
Therapy attendance shows the strongest negative correlation with anxiety scores among High-Anxiety individuals
Substance use is disproportionately elevated in the High-Anxiety group
Meditation frequency shows a statistically significant reduction in anxiety in the Low-Anxiety population but attenuated effects in severe cases
Gradient Boosting SHAP values confirm therapy and substance use as top-2 predictive features
---
🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/jarvissimms12/Modeling-Anxiety-and-Depression.git
cd Modeling-Anxiety-and-Depression

# 2. Install dependencies
pip install pandas seaborn matplotlib scikit-learn shap

# 3. Run the analysis scripts
python "Coping Strategies.py"       # Population segmentation + visualizations
python gradient_boosting_&_shap.py  # Predictive model + SHAP explainability
```
The dataset (`anxiety_depression_data.csv`) is included — no external download needed.
---
📁 Project Structure
```
Modeling-Anxiety-and-Depression/
├── Coping Strategies.py              # Segmentation analysis & visualizations
├── gradient_boosting_&_shap.py       # GBM model + SHAP feature importance
├── anxiety_depression_data.csv       # Dataset (included)
└── README.md
```
---
🔑 Key Skills Demonstrated
Exploratory Data Analysis (EDA) — distribution analysis, group comparisons
Statistical segmentation — median-based population splits
Machine learning — Gradient Boosting classification
Model explainability — SHAP values for feature attribution
Data visualization — seaborn grouped bar charts, SHAP summary plots
Domain translation — communicating technical findings in plain language
---
🔗 Related Projects
Media Content Analytics Dashboard — Interactive BI dashboard with Streamlit & Plotly
NYC Housing Data Analysis — Real-world data pipeline with NYC Open Data API
Credit Card Fraud Detection — Imbalanced classification with SMOTE + Random Forest
---
👤 Author
Jarvis Simms | MS Data Science, NYIT 2025 | Brooklyn, NY  
GitHub • LinkedIn
