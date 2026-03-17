# HR Turnover Retention Dashboard

This project delivers a responsible HR AI dashboard that identifies the **top 5 active employees most likely to resign**, estimates a **probable reason category**, and suggests **recommended HR actions** to reduce attrition risk.

The solution is intentionally aligned with three hackathon themes:

- `Explainable AI`
- `Ethical AI`
- `Frugal AI` as a light bonus

## Product Goal

For active employees only, the dashboard must answer three questions:

1. **Who is most at risk of voluntary turnover?**
2. **Why are they likely to leave?**
3. **What should HR do now to retain them?**

The final output is a **Streamlit dashboard** designed for an HR Business Partner.

## Use Case

Help HR prioritize retention actions by ranking active employees by turnover risk and attaching an interpretable explanation, a probable reason, and an action plan.

## Persona

**HR Business Partner**

Needs:
- a ranked list of employees to review first;
- understandable reasons behind each risk score;
- actionable retention recommendations;
- a solution that avoids training directly on sensitive attributes.

## What the Dashboard Shows

### 1. Top 5 employees at risk

The dashboard scores all active employees and returns the **top 5 highest predicted turnover risks**.

For each employee, it shows:
- anonymized employee alias;
- department and position;
- turnover risk score;
- probable reason category;
- closest historical resignation reasons;
- recommended HR actions.

### 2. Probable reason engine

The dashboard does **not** claim to know the true resignation motive.

Instead, it estimates a **probable reason category** using:
- similarity to historical voluntary leavers;
- current HR features such as tenure, engagement, absences, salary positioning, and recruitment source.

Reason categories used in the dashboard:
- `Compensation/Career`
- `Work Conditions/Engagement`
- `Personal/External`

This is presented as a **decision-support hypothesis**, not as a certainty.

### 3. Recommended actions

The dashboard converts the probable reason and the main risk drivers into concrete HR recommendations such as:
- salary benchmark review;
- internal mobility discussion;
- workload and schedule review;
- manager check-in;
- mentoring and onboarding support;
- stretch project assignment;
- flexibility or accommodation discussion.

## Responsible AI Positioning

### Explainable AI

- The primary risk model is **Logistic Regression**.
- Each selected employee is accompanied by main risk drivers.
- The reason engine is based on similar historical leavers and visible business rules.

### Ethical AI

- Employee identities are anonymized into aliases like `EMP_001`.
- Sensitive attributes are **not used for training**:
  - `Sex`
  - `RaceDesc`
  - `GenderID`
  - `CitizenDesc`
  - `HispanicLatino`
- Sensitive attributes are kept only for post-model fairness checks.

### Frugal AI

- A lightweight **Logistic Regression** model is benchmarked against **Random Forest**.
- Logistic Regression is retained because it is strong, fast, lightweight, and easier to govern.

## Data Card

| Item | Summary |
| --- | --- |
| Dataset purpose | Understand voluntary employee turnover and support retention actions |
| Source | Synthetic HR dataset inspired by the Rich Huebner case study |
| Population used | Active employees + voluntary leavers only |
| Rows after filtering | 293 |
| Prediction target | `Termd = 1` means voluntary turnover |
| Active employees scored in dashboard | 207 |
| Sensitive fields in raw dataset | `Sex`, `RaceDesc`, `GenderID`, `CitizenDesc`, `HispanicLatino` |
| Direct identifiers in raw dataset | `Employee_Name`, `EmpID`, `ManagerName` |
| Identifier handling | Replaced at load time by `Employee_Alias` |
| Leakage columns removed from modeling | `TermReason`, `DateofTermination`, `EmploymentStatus`, `EmpStatusID` |
| Historical time features engineered | `AgeAtReview`, `TenureAtReview` |

## Model Card

### Risk model

| Item | Summary |
| --- | --- |
| Primary model | Logistic Regression |
| Benchmark comparator | Random Forest |
| Training target | Voluntary turnover risk (`Termd`) |
| Features used | Role, department, salary, recruitment source, performance, engagement, satisfaction, projects, lateness, absences, age at review, tenure at review, state |
| Sensitive features in training | None |
| Output | Probability of voluntary turnover |
| Decision role | HR decision support only |

### Reason engine

| Item | Summary |
| --- | --- |
| Engine type | Similarity-based inference over historical voluntary leavers |
| Output | Probable reason category + closest historical raw reasons |
| Why not a hard multiclass model | The voluntary leaver sample is small and imbalanced, so the dashboard uses a more cautious case-based estimation |
| Governance position | Reason is a hypothesis for HR review, not a fact |

## Results Summary

### Risk model benchmark

| Model | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9398 | 0.8996 | 0.9205 | 0.8846 | 0.8462 | 0.8679 |
| Random Forest | 0.9516 | 0.9219 | 0.8977 | 0.9474 | 0.6923 | 0.8000 |

Final choice: **Logistic Regression**

Reason:
- nearly the same ranking quality as Random Forest;
- better recall/F1 balance for retention use;
- much simpler to explain in a jury setting;
- more frugal and easier to deploy.

## Fairness Snapshot

The dashboard includes post-model fairness views by `Sex` and `RaceDesc`.

Important caution:
- some groups are small;
- these are governance indicators, not causal claims;
- the fairness section is meant to support responsible review, not to certify fairness.

## Main Risk Patterns Found

Across the model and the employee-level explanations, the strongest recurring patterns are:

- very short tenure is a major risk signal;
- some external recruitment sources are associated with higher turnover;
- some operational roles, especially in Production, appear more exposed;
- lower engagement and lower satisfaction increase risk;
- lack of special projects can signal lower development opportunities;
- high absences or lateness can indicate disengagement or work-life constraints.

## Repository Files

- `app.py`: Streamlit dashboard
- `notebooks/eda.ipynb`: full exploratory and modeling notebook
- `data/HRDataset_v14.csv`: source dataset
- `README.md`: project summary
- `requirements.txt`: project dependencies

## How to Run

From the project directory:

```bash
cd "/Users/alexandrefau/ESILV/A4/explainibility AI/Hackathon_AI_explainability-"
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

If you already attempted an install with the old `requirements.txt`, the cleanest recovery path is to recreate the virtual environment and reinstall:

```bash
cd "/Users/alexandrefau/ESILV/A4/explainibility AI/Hackathon_AI_explainability-"
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Demo Flow

The dashboard is designed for a short oral defense:

1. Show the benchmark and justify the choice of Logistic Regression.
2. Show the top 5 active employees at risk.
3. Open one employee profile and explain:
   - the risk score;
   - the probable reason category;
   - the closest historical resignation reasons;
   - the recommended actions.
4. Finish with the fairness tab and the responsible AI positioning.

## Limits

- The dataset is synthetic and relatively small.
- The dashboard predicts **risk**, not causality.
- The probable reason is an informed hypothesis, not ground truth.
- Some fairness groups are under-represented.
- The current version does not yet integrate text feedback or NLP.

## Pitch

We built a responsible HR AI dashboard that helps HR act before employees leave. The system ranks the top 5 active employees most likely to resign, estimates a probable reason by comparing them to similar historical leavers, and translates the result into concrete retention actions. We deliberately chose a Logistic Regression model because it stays highly competitive while remaining transparent, lightweight, and easy to defend. Sensitive attributes are excluded from training, fairness is audited separately, and all outputs are framed as decision support rather than automated HR decisions.
