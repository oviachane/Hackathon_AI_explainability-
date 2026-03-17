# Hackathon AI Explainability

## Group & Members

Group 23

* Ovia CHANEMOUGANANDA
* Noa LIEGEOIS
* Alexandre FAU
* Marie-Lou JODET
* Arshitha KANDUKURI

---

## Use Case

This project aims to support HR teams in identifying employees at risk of voluntary turnover, understanding the key drivers behind that risk, and taking appropriate retention actions.

The tool focuses on three main objectives:

* predicting resignation risk,
* explaining the factors contributing to that risk,
* providing actionable recommendations for HR.

---

## Persona

The primary user of this tool is an HR Business Partner who needs an interpretable decision-support system to prioritize retention efforts.

---

## Summary

This repository contains two main components:

* `notebooks/eda.ipynb`: data exploration, preprocessing, and feature analysis
* `app.py`: a Streamlit dashboard for risk prediction and HR decision support

---

## Dataset and Target

The dataset includes HR-related information such as:

* job and department data
* salary and performance indicators
* engagement and satisfaction scores
* attendance and absenteeism

### Target Variable

* `Termd = 0`: active employee
* `Termd = 1`: employee who left the company

The analysis focuses on voluntary turnover only.

---

## Modeling Approach

The model used is a Logistic Regression, chosen for its interpretability.

* Type: binary classification
* Output: probability of resignation

### Data preparation:

* filtering to voluntary departures
* handling missing values using imputation
* feature engineering 
* selection of relevant variables based on exploratory analysis


---

## Dashboard Functionality

The Streamlit dashboard:

* filters the dataset to voluntary turnover
* predicts resignation risk for active employees
* identifies a shortlist of employees with the highest risk
* estimates a likely resignation reason
* suggests HR retention actions
* anonymizes employee identities

---

## How to Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

---

## Responsible AI

The project follows basic ethical AI principles:

* exclusion of sensitive attributes such as gender, ethnicity, citizenship, and marital status
* anonymization of employee identities
* use as a decision-support tool not an automated decision system

---

## Business Value

This solution enables HR teams to:

* anticipate turnover risk
* better understand its drivers
* take targeted and proactive actions

It supports a shift from reactive to proactive HR management.

---

## Conclusion

This project illustrates how explainable AI can be applied to HR use cases to generate interpretable insights and support better decision-making.
