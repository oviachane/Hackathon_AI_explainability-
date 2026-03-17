# Employee Turnover Risk Prediction for HR Decision Support

## Project Overview

This project was developed during a hackathon focused on building an AI solution for human resources.

Our objective is to help HR teams identify employees who are at risk of leaving the company and understand the main factors behind this risk.

The final solution combines:
- a predictive model for employee turnover risk,
- explainable AI techniques,
- an HR-oriented dashboard for decision support.

## Problem Statement

Employee turnover can generate major costs for companies, including:
- recruitment costs,
- onboarding time,
- productivity loss,
- knowledge loss within teams.

Our goal is to predict voluntary turnover risk and provide interpretable insights so that HR teams can take preventive action.

## Main Objectives

- Predict the probability that an employee will leave the company
- Identify the most at-risk active employees
- Explain the main factors driving turnover risk
- Support HR decision-making with actionable recommendations

## AI Themes Addressed

This project focuses on two key AI themes:

### 1. Explainable AI
We selected an interpretable modeling approach and used SHAP explanations to understand the drivers behind each prediction.

### 2. Ethical AI
We paid particular attention to the use of sensitive variables and positioned the model as a decision-support tool rather than an automated decision-maker.

## Dataset

The dataset contains employee HR information such as:
- demographics,
- job information,
- salary,
- performance indicators,
- absenteeism,
- engagement and satisfaction scores,
- employment status and termination information.

### Target Variable

The target variable is:

- `Termd = 0` → employee still active
- `Termd = 1` → employee left the company

To align the project with voluntary turnover prediction, involuntary departures were filtered out before modeling.

## Methodology

### 1. Data Cleaning and Filtering
- removed involuntary departures,
- kept active employees and employee-driven departures,
- handled missing values,
- created additional variables such as age and tenure.

### 2. Exploratory Data Analysis
We explored the relationship between turnover and multiple HR-related factors using:
- descriptive statistics,
- correlation analysis,
- cleaned correlation with turnover,
- feature selection based on interpretable criteria.

### 3. Predictive Modeling
We trained a classification model to estimate the probability of employee departure.

Current model:
- Logistic Regression
- missing values handled with median imputation

### 4. Explainability
We used SHAP to:
- measure global feature importance,
- explain individual predictions,
- make model outputs understandable for HR teams.

### 5. Business Use Case
The model is applied to active employees only in order to identify the employees currently most at risk of leaving.

## Key Outputs

The project produces:
- turnover risk scores,
- a top 5 list of active employees at highest risk,
- the main explanatory factors behind the predictions,
- recommendations for HR action.

## Dashboard Objective

The dashboard is designed for HR use and aims to answer three questions:

1. Who is at risk of leaving?
2. Why are they at risk?
3. What can HR do to reduce this risk?

## Repository Structure

```text
.
├── data/
│   └── HRDataset_v14.csv
├── notebooks/
│   └── eda.ipynb
├── src/
├── app.py
├── requirements.txt
└── README.md
