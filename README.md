# Hackathon AI Explainability

This repository now contains two main parts:

- the notebook `notebooks/eda.ipynb` for HR data exploration,
- a Streamlit dashboard `app.py` to prioritize active employees with the highest resignation risk.

## What the dashboard does

- filters the dataset to voluntary turnover only,
- trains an explainable model (`LogisticRegression`) to estimate resignation risk,
- produces a priority shortlist of active employees,
- estimates a likely resignation reason,
- proposes concrete HR retention actions for each profile,
- permanently hides employee names to stay aligned with the hackathon anonymization constraint.

## Run the dashboard

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Method

- Business scope: voluntary exits only.
- Sensitive variables excluded from scoring: sex, ethnicity, citizenship, marital status, and similar fields.
- Final displayed score: `70% model score + 30% actionable HR signals`.
- Likely reason: combination of historical similarity with past voluntary exits and HR-oriented business rules.
