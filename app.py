from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "HRDataset_v14.csv"
TARGET = "Termd"
TOP_N = 5

VOLUNTARY_REASONS = [
    "Another position",
    "unhappy",
    "more money",
    "career change",
    "hours",
    "return to school",
    "relocation out of area",
    "maternity leave - did not return",
    "retiring",
    "military",
    "medical issues",
]

RISK_FEATURES = [
    "MaritalDesc",
    "Salary",
    "Position",
    "Department",
    "RecruitmentSource",
    "PerformanceScore",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "FromDiversityJobFairID",
    "AgeAtReview",
    "TenureAtReview",
    "State",
]

REASON_GROUP_MAP = {
    "Another position": "Compensation/Career",
    "more money": "Compensation/Career",
    "career change": "Compensation/Career",
    "unhappy": "Work Conditions/Engagement",
    "hours": "Work Conditions/Engagement",
    "return to school": "Personal/External",
    "maternity leave - did not return": "Personal/External",
    "relocation out of area": "Personal/External",
    "military": "Personal/External",
    "medical issues": "Personal/External",
    "retiring": "Personal/External",
}

REASON_DISPLAY_MAP = {
    "Compensation/Career": "Compensation or career",
    "Work Conditions/Engagement": "Work conditions or engagement",
    "Personal/External": "Personal or external factors",
}


def anonymize_employee_names(frame: pd.DataFrame) -> pd.DataFrame:
    anonymized = frame.copy()
    anonymized["Employee_Alias"] = [f"EMP_{i + 1:03d}" for i in range(len(anonymized))]
    return anonymized


def parse_dob_with_century_fix(series: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    parsed.loc[parsed > reference_date] = parsed.loc[parsed > reference_date] - pd.DateOffset(years=100)
    return parsed


def build_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    engineered["ReviewDate"] = pd.to_datetime(engineered["LastPerformanceReview_Date"], errors="coerce")
    engineered["HireDateParsed"] = pd.to_datetime(engineered["DateofHire"], errors="coerce")
    engineered["DOBParsed"] = parse_dob_with_century_fix(engineered["DOB"], engineered["ReviewDate"].max())
    engineered["AgeAtReview"] = ((engineered["ReviewDate"] - engineered["DOBParsed"]).dt.days / 365.25).round(1)
    engineered["TenureAtReview"] = (
        (engineered["ReviewDate"] - engineered["HireDateParsed"]).dt.days / 365.25
    ).round(1)
    return engineered


def make_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in feature_frame.columns if col not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    total = sum(score_map.values())
    if total == 0:
        return {key: 0.0 for key in score_map}
    return {key: value / total for key, value in score_map.items()}


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame = anonymize_employee_names(frame)
    frame = frame[
        (frame["TermReason"] == "N/A-StillEmployed") | (frame["TermReason"].isin(VOLUNTARY_REASONS))
    ].copy()
    frame = build_historical_features(frame)
    return frame


def compute_metrics(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
    }


def fairness_table(frame: pd.DataFrame, sensitive_column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(sensitive_column)
        .agg(
            n=(TARGET, "size"),
            actual_turnover=(TARGET, "mean"),
            predicted_positive=("prediction", "mean"),
            avg_score=("score", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
        .round(4)
    )
    return summary


def compute_reason_distribution(
    row: pd.Series,
    transformed_row,
    leavers: pd.DataFrame,
    nn_model: NearestNeighbors,
    stats: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    distances, indices = nn_model.kneighbors(transformed_row, return_distance=True)
    distances = distances[0]
    indices = indices[0]
    weights = 1.0 / (distances + 1e-6)

    group_scores = {
        "Compensation/Career": 0.0,
        "Work Conditions/Engagement": 0.0,
        "Personal/External": 0.0,
    }
    raw_scores: Dict[str, float] = {}
    raw_votes: Dict[str, int] = {}

    for neighbor_idx, weight in zip(indices, weights):
        neighbor = leavers.iloc[neighbor_idx]
        reason_group = neighbor["ReasonGroup"]
        raw_reason = neighbor["TermReason"]
        group_scores[reason_group] = group_scores.get(reason_group, 0.0) + float(weight)
        raw_scores[raw_reason] = raw_scores.get(raw_reason, 0.0) + float(weight)
        raw_votes[raw_reason] = raw_votes.get(raw_reason, 0) + 1

    position_salary_median = stats["position_salary_median"].get(row["Position"], row["Salary"])
    overall_turnover = stats["overall_turnover_rate"]
    recruitment_risk = stats["recruitment_turnover"].get(row["RecruitmentSource"], overall_turnover)

    if row["TenureAtReview"] <= 2:
        group_scores["Compensation/Career"] += 0.12
    if row["Salary"] < position_salary_median * 0.95:
        group_scores["Compensation/Career"] += 0.10
    if recruitment_risk > overall_turnover + 0.05:
        group_scores["Compensation/Career"] += 0.06

    if row["EngagementSurvey"] < 4.0:
        group_scores["Work Conditions/Engagement"] += 0.10
    if row["EmpSatisfaction"] <= 3:
        group_scores["Work Conditions/Engagement"] += 0.12
    if row["Absences"] >= stats["high_absence_threshold"] or row["DaysLateLast30"] >= 1:
        group_scores["Work Conditions/Engagement"] += 0.08

    if row["AgeAtReview"] >= 50:
        group_scores["Personal/External"] += 0.07
    if row["State"] != "MA":
        group_scores["Personal/External"] += 0.05

    return normalize_scores(group_scores), normalize_scores(raw_scores), raw_votes


def build_risk_driver_messages(
    row: pd.Series,
    contribution_map: Dict[str, float],
    stats: Dict[str, Dict[str, float]],
) -> List[Tuple[str, float]]:
    candidates: List[Tuple[str, float]] = []
    overall_turnover = stats["overall_turnover_rate"]
    department_turnover = stats["department_turnover"].get(row["Department"], overall_turnover)
    recruitment_turnover = stats["recruitment_turnover"].get(row["RecruitmentSource"], overall_turnover)
    position_salary_median = stats["position_salary_median"].get(row["Position"], row["Salary"])

    if row["TenureAtReview"] <= 2:
        candidates.append(
            (
                f"Short tenure: only {row['TenureAtReview']:.1f} year(s) since hire, which is strongly linked to early turnover in this dataset.",
                abs(contribution_map.get("num__TenureAtReview", 0.0)) + 0.20,
            )
        )
    if recruitment_turnover > overall_turnover + 0.05:
        feature_key = f"cat__RecruitmentSource_{row['RecruitmentSource']}"
        candidates.append(
            (
                f"Recruitment source '{row['RecruitmentSource']}' has a higher-than-average turnover pattern in this dataset.",
                max(contribution_map.get(feature_key, 0.0), 0.10),
            )
        )
    if department_turnover > overall_turnover + 0.05:
        feature_key = f"cat__Department_{row['Department']}"
        candidates.append(
            (
                f"Department '{row['Department']}' shows above-average voluntary turnover in historical data.",
                max(contribution_map.get(feature_key, 0.0), 0.10),
            )
        )
    if row["EngagementSurvey"] < 4.0:
        candidates.append(
            (
                f"Engagement score is below the safer range ({row['EngagementSurvey']:.1f}/5).",
                max(contribution_map.get("num__EngagementSurvey", 0.0), 0.08),
            )
        )
    if row["EmpSatisfaction"] <= 3:
        candidates.append(
            (
                f"Employee satisfaction is only {int(row['EmpSatisfaction'])}/5.",
                max(contribution_map.get("num__EmpSatisfaction", 0.0), 0.08),
            )
        )
    if row["SpecialProjectsCount"] == 0:
        candidates.append(
            (
                "No special projects are currently assigned, which can signal limited development opportunities.",
                max(contribution_map.get("num__SpecialProjectsCount", 0.0), 0.06),
            )
        )
    if row["Absences"] >= stats["high_absence_threshold"]:
        candidates.append(
            (
                f"Absence count is elevated ({int(row['Absences'])}).",
                max(contribution_map.get("num__Absences", 0.0), 0.06),
            )
        )
    if row["DaysLateLast30"] >= 1:
        candidates.append(
            (
                f"Recent lateness detected ({int(row['DaysLateLast30'])} day(s) late in the last 30 days).",
                max(contribution_map.get("num__DaysLateLast30", 0.0), 0.06),
            )
        )
    if row["Salary"] < position_salary_median * 0.95:
        candidates.append(
            (
                f"Salary is below the median for the same role ({row['Salary']:.0f} vs {position_salary_median:.0f}).",
                max(contribution_map.get("num__Salary", 0.0), 0.06),
            )
        )

    candidates.sort(key=lambda item: item[1], reverse=True)

    unique_messages: List[Tuple[str, float]] = []
    seen = set()
    for message, score in candidates:
        if message not in seen:
            unique_messages.append((message, score))
            seen.add(message)

    return unique_messages[:4]


def build_recommended_actions(
    row: pd.Series,
    probable_reason_group: str,
    top_raw_reasons: List[str],
    stats: Dict[str, Dict[str, float]],
) -> List[str]:
    actions: List[str] = []
    role_salary_median = stats["position_salary_median"].get(row["Position"], row["Salary"])

    if probable_reason_group == "Compensation/Career":
        actions.extend(
            [
                "Run a salary and market benchmark review for this role.",
                "Schedule a career-path or internal mobility conversation within the next 2 weeks.",
                "Offer a concrete development plan with progression milestones.",
            ]
        )
    elif probable_reason_group == "Work Conditions/Engagement":
        actions.extend(
            [
                "Launch an HR and manager check-in focused on satisfaction, workload, and team conditions.",
                "Review schedule flexibility, workload balance, and manager support.",
                "Re-measure engagement after the action plan is applied.",
            ]
        )
    else:
        actions.extend(
            [
                "Discuss personal constraints confidentially and explore flexibility options.",
                "Review internal transfer, remote work, or leave accommodations if relevant.",
                "Prepare a retention and continuity plan with HR and the manager.",
            ]
        )

    if "more money" in top_raw_reasons or row["Salary"] < role_salary_median * 0.95:
        actions.append("Validate whether compensation is below internal benchmarks and decide on remediation.")
    if "Another position" in top_raw_reasons or "career change" in top_raw_reasons:
        actions.append("Propose a visible internal mobility path before the employee looks outside.")
    if "hours" in top_raw_reasons or row["DaysLateLast30"] >= 1 or row["Absences"] >= stats["high_absence_threshold"]:
        actions.append("Review workload, planning, and work-life balance constraints with the employee.")
    if "unhappy" in top_raw_reasons or row["EngagementSurvey"] < 4.0 or row["EmpSatisfaction"] <= 3:
        actions.append("Hold a manager-quality and team-experience review with a short HR follow-up.")
    if row["TenureAtReview"] <= 2:
        actions.append("Assign a mentor and set a 30-60-90 day retention checkpoint.")
    if row["SpecialProjectsCount"] == 0:
        actions.append("Offer a stretch project or cross-functional assignment to increase engagement.")

    deduplicated: List[str] = []
    seen = set()
    for action in actions:
        if action not in seen:
            deduplicated.append(action)
            seen.add(action)

    return deduplicated[:5]


@st.cache_resource(show_spinner=False)
def build_dashboard_assets():
    df = load_dataset()
    overall_turnover_rate = float(df[TARGET].mean())
    stats = {
        "overall_turnover_rate": overall_turnover_rate,
        "department_turnover": df.groupby("Department")[TARGET].mean().to_dict(),
        "recruitment_turnover": df.groupby("RecruitmentSource")[TARGET].mean().to_dict(),
        "position_salary_median": df.groupby("Position")["Salary"].median().to_dict(),
        "high_absence_threshold": float(df["Absences"].quantile(0.75)),
    }

    X = df[RISK_FEATURES].copy()
    y = df[TARGET].astype(int)
    audit = df[["Sex", "RaceDesc", TARGET]].copy()

    X_train, X_test, y_train, y_test, audit_train, audit_test = train_test_split(
        X,
        y,
        audit,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    eval_preprocessor = make_preprocessor(X_train)
    eval_logreg = Pipeline(
        steps=[
            ("preprocessor", eval_preprocessor),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )
    eval_rf = Pipeline(
        steps=[
            ("preprocessor", eval_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

    eval_logreg.fit(X_train, y_train)
    logreg_scores = eval_logreg.predict_proba(X_test)[:, 1]
    logreg_predictions = (logreg_scores >= 0.5).astype(int)
    logreg_metrics = compute_metrics(y_test, logreg_scores, logreg_predictions)

    eval_rf.fit(X_train, y_train)
    rf_scores = eval_rf.predict_proba(X_test)[:, 1]
    rf_predictions = (rf_scores >= 0.5).astype(int)
    rf_metrics = compute_metrics(y_test, rf_scores, rf_predictions)

    fairness_input = audit_test.copy()
    fairness_input["score"] = logreg_scores
    fairness_input["prediction"] = logreg_predictions
    sex_fairness = fairness_table(fairness_input, "Sex")
    race_fairness = fairness_table(fairness_input, "RaceDesc")

    deployment_preprocessor = make_preprocessor(X)
    deployment_risk_model = Pipeline(
        steps=[
            ("preprocessor", deployment_preprocessor),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )
    deployment_risk_model.fit(X, y)

    feature_names = deployment_risk_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.Series(deployment_risk_model.named_steps["model"].coef_[0], index=feature_names)

    leavers = df[df[TARGET] == 1].copy()
    leavers["ReasonGroup"] = leavers["TermReason"].map(REASON_GROUP_MAP)
    leaver_vectors = deployment_risk_model.named_steps["preprocessor"].transform(leavers[RISK_FEATURES])
    nn_model = NearestNeighbors(n_neighbors=min(7, len(leavers)), metric="cosine")
    nn_model.fit(leaver_vectors)

    active = df[df[TARGET] == 0].copy()
    active_scores = deployment_risk_model.predict_proba(active[RISK_FEATURES])[:, 1]
    active["risk_score"] = active_scores

    active_vectors = deployment_risk_model.named_steps["preprocessor"].transform(active[RISK_FEATURES])
    if hasattr(active_vectors, "toarray"):
        active_vectors_dense = active_vectors.toarray()
    else:
        active_vectors_dense = active_vectors

    contribution_frame = pd.DataFrame(
        active_vectors_dense * coefficients.values,
        index=active.index,
        columns=feature_names,
    )

    enriched_rows = []
    for idx, row in active.iterrows():
        transformed_row = deployment_risk_model.named_steps["preprocessor"].transform(row[RISK_FEATURES].to_frame().T)
        group_probs, raw_probs, raw_votes = compute_reason_distribution(
            row,
            transformed_row,
            leavers,
            nn_model,
            stats,
        )
        probable_reason_group = max(group_probs, key=group_probs.get)
        top_raw_reasons = list(sorted(raw_probs, key=raw_probs.get, reverse=True)[:2])
        contribution_map = contribution_frame.loc[idx].to_dict()
        driver_messages = build_risk_driver_messages(row, contribution_map, stats)
        actions = build_recommended_actions(row, probable_reason_group, top_raw_reasons, stats)

        enriched_rows.append(
            {
                "Employee_Alias": row["Employee_Alias"],
                "EmpID": int(row["EmpID"]),
                "Department": row["Department"].strip(),
                "Position": row["Position"],
                "RecruitmentSource": row["RecruitmentSource"],
                "risk_score": float(row["risk_score"]),
                "probable_reason_group": probable_reason_group,
                "reason_confidence": float(group_probs[probable_reason_group]),
                "similar_raw_reasons": ", ".join(top_raw_reasons),
                "reason_distribution": group_probs,
                "raw_reason_distribution": raw_probs,
                "raw_reason_votes": raw_votes,
                "risk_drivers": [message for message, _ in driver_messages],
                "recommended_actions": actions,
                "Salary": row["Salary"],
                "EngagementSurvey": row["EngagementSurvey"],
                "EmpSatisfaction": row["EmpSatisfaction"],
                "SpecialProjectsCount": row["SpecialProjectsCount"],
                "DaysLateLast30": row["DaysLateLast30"],
                "Absences": row["Absences"],
                "AgeAtReview": row["AgeAtReview"],
                "TenureAtReview": row["TenureAtReview"],
            }
        )

    leaderboard = pd.DataFrame(enriched_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    top5 = leaderboard.head(TOP_N).copy()

    benchmark = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "ROC-AUC": round(logreg_metrics["roc_auc"], 4),
                "Average Precision": round(logreg_metrics["average_precision"], 4),
                "Accuracy": round(logreg_metrics["accuracy"], 4),
                "Precision": round(logreg_metrics["precision"], 4),
                "Recall": round(logreg_metrics["recall"], 4),
                "F1": round(logreg_metrics["f1"], 4),
            },
            {
                "Model": "Random Forest",
                "ROC-AUC": round(rf_metrics["roc_auc"], 4),
                "Average Precision": round(rf_metrics["average_precision"], 4),
                "Accuracy": round(rf_metrics["accuracy"], 4),
                "Precision": round(rf_metrics["precision"], 4),
                "Recall": round(rf_metrics["recall"], 4),
                "F1": round(rf_metrics["f1"], 4),
            },
        ]
    )

    return {
        "dataset": df,
        "benchmark": benchmark,
        "sex_fairness": sex_fairness,
        "race_fairness": race_fairness,
        "leaderboard": leaderboard,
        "top5": top5,
        "stats": stats,
    }


def build_top5_table(top5: pd.DataFrame) -> pd.DataFrame:
    display = top5.reset_index(drop=True).copy()
    display["Priority"] = display.index + 1
    display["Employee"] = display.apply(format_employee_label, axis=1)
    display["Risk"] = display["risk_score"].map(lambda x: f"{x:.0%}")
    display["Priority level"] = display["risk_score"].map(risk_priority_label)
    display["Likely reason"] = display["probable_reason_group"].map(reason_display_label)
    display["First HR action"] = display["recommended_actions"].map(
        lambda actions: shorten_text(actions[0], 90) if actions else ""
    )
    display = display.rename(columns={"Position": "Role"})
    return display[
        [
            "Priority",
            "Employee",
            "Department",
            "Role",
            "Risk",
            "Priority level",
            "Likely reason",
            "First HR action",
        ]
    ]


def reason_display_label(reason_group: str) -> str:
    return REASON_DISPLAY_MAP.get(reason_group, reason_group)


def risk_priority_label(score: float) -> str:
    if score >= 0.75:
        return "Immediate action"
    if score >= 0.55:
        return "High priority"
    return "Monitor"


def format_employee_label(row: pd.Series) -> str:
    return f"{row['Employee_Alias']} (EmpID: {int(row['EmpID'])})"


def top_theme_label(top5: pd.DataFrame) -> str:
    if top5.empty:
        return "N/A"
    return reason_display_label(top5["probable_reason_group"].mode().iloc[0])


def shorten_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def reason_distribution_chart(reason_distribution: Dict[str, float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Reason group": [reason_display_label(reason) for reason in reason_distribution.keys()],
            "Probability": list(reason_distribution.values()),
        }
    )
    figure = px.bar(
        frame.sort_values("Probability", ascending=True),
        x="Probability",
        y="Reason group",
        orientation="h",
        text=frame.sort_values("Probability", ascending=True)["Probability"].map(lambda x: f"{x:.0%}"),
        color="Probability",
        color_continuous_scale="Blues",
    )
    figure.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    figure.update_traces(textposition="outside")
    return figure


def risk_gauge(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444"},
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": "Risk of leaving"},
        )
    )
    figure.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return figure


def fairness_chart(frame: pd.DataFrame, category_column: str, value_column: str, title: str) -> go.Figure:
    figure = px.bar(
        frame,
        x=category_column,
        y=value_column,
        text=frame[value_column].map(lambda x: f"{x:.0%}"),
        color=value_column,
        color_continuous_scale="Oranges",
    )
    figure.update_layout(height=320, title=title, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
    figure.update_traces(textposition="outside")
    return figure


def main():
    st.set_page_config(page_title="HR Turnover Retention Dashboard", layout="wide")
    st.title("HR Turnover Retention Dashboard")
    st.caption(
        "Quick HR view: identify who needs attention first, why they may leave, and what to do now."
    )

    assets = build_dashboard_assets()
    dataset = assets["dataset"]
    benchmark = assets["benchmark"]
    sex_fairness = assets["sex_fairness"]
    race_fairness = assets["race_fairness"]
    leaderboard = assets["leaderboard"]
    top5 = assets["top5"]

    avg_top5_risk = top5["risk_score"].mean()
    highest_risk = float(top5["risk_score"].max()) if not top5.empty else 0.0

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Active employees reviewed", int((dataset[TARGET] == 0).sum()))
    metric_col_2.metric("Employees to review now", len(top5))
    metric_col_3.metric("Highest risk in Top 5", f"{highest_risk:.0%}")
    metric_col_4.metric("Most common risk theme", top_theme_label(top5))

    st.info(
        "Reading guide for HR: start with the Top 5 priority list, then open one employee sheet to review "
        "the likely reason and the recommended retention actions."
    )

    overview_tab, employee_tab, responsible_ai_tab = st.tabs(["HR Priority View", "Employee Sheet", "Responsible AI"])

    with overview_tab:
        st.subheader("Top 5 priority list")
        st.dataframe(build_top5_table(top5), use_container_width=True, hide_index=True)
        st.caption(
            "The risk score is a probability of voluntary turnover. It is used to prioritize HR follow-up, "
            "not to automate decisions."
        )

        st.subheader("Action cards")
        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            expander_title = (
                f"Priority {rank} | {format_employee_label(row)} | "
                f"Risk {row['risk_score']:.0%} | {reason_display_label(row['probable_reason_group'])}"
            )
            with st.expander(expander_title, expanded=rank == 1):
                top_cols = st.columns(3)
                top_cols[0].metric("Risk", f"{row['risk_score']:.0%}")
                top_cols[1].metric("Priority level", risk_priority_label(float(row["risk_score"])))
                top_cols[2].metric("Likely reason", reason_display_label(row["probable_reason_group"]))

                why_col, actions_col = st.columns(2)
                with why_col:
                    st.markdown("**Why this employee is flagged**")
                    for driver in row["risk_drivers"][:3]:
                        st.markdown(f"- {driver}")
                with actions_col:
                    st.markdown("**What HR should do now**")
                    for action in row["recommended_actions"][:3]:
                        st.markdown(f"- {action}")

                st.caption(
                    f"Closest historical reasons: {row['similar_raw_reasons']}. "
                    f"Reason confidence: {row['reason_confidence']:.0%}."
                )

        with st.expander("Technical view: model comparison and overall score distribution"):
            tech_left, tech_right = st.columns([1.0, 1.2])
            with tech_left:
                st.subheader("Model comparison used for deployment choice")
                st.dataframe(benchmark, use_container_width=True, hide_index=True)
            with tech_right:
                risk_histogram = px.histogram(
                    leaderboard,
                    x="risk_score",
                    nbins=20,
                    title="Distribution of turnover risk across active employees",
                    labels={"risk_score": "Predicted risk"},
                    color_discrete_sequence=["#2563eb"],
                )
                risk_histogram.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(risk_histogram, use_container_width=True)

    with employee_tab:
        st.subheader("Employee retention sheet")

        employee_options = {
            row["Employee_Alias"]: (
                f"{format_employee_label(row)}"
                f" - {row['Position']} ({row['Department']})"
            )
            for _, row in top5.iterrows()
        }
        selected_alias = st.selectbox(
            "Select one employee from the Top 5",
            options=list(employee_options.keys()),
            format_func=lambda alias: employee_options[alias],
        )
        selected = top5[top5["Employee_Alias"] == selected_alias].iloc[0]

        metric_left, metric_mid, metric_right = st.columns(3)
        metric_left.metric("Risk", f"{selected['risk_score']:.0%}")
        metric_mid.metric("Priority level", risk_priority_label(float(selected["risk_score"])))
        metric_right.metric("Likely reason", reason_display_label(selected["probable_reason_group"]))

        top_left, top_right = st.columns([0.8, 1.2])
        with top_left:
            st.plotly_chart(risk_gauge(float(selected["risk_score"])), use_container_width=True)
        with top_right:
            st.subheader("Quick profile")
            st.markdown(f"**Employee:** {format_employee_label(selected)}")
            st.markdown(f"**Role:** {selected['Position']} | **Department:** {selected['Department']}")
            st.markdown(f"**Recruitment source:** {selected['RecruitmentSource']}")
            st.markdown(
                f"**Tenure:** {selected['TenureAtReview']:.1f} years | "
                f"**Engagement:** {selected['EngagementSurvey']:.1f}/5 | "
                f"**Satisfaction:** {int(selected['EmpSatisfaction'])}/5"
            )
            st.markdown(f"**Likely reason:** {reason_display_label(selected['probable_reason_group'])}")
            st.markdown(f"**Confidence in this reason:** {selected['reason_confidence']:.0%}")
            st.markdown(f"**Closest historical reasons:** {selected['similar_raw_reasons']}")

        reason_col, actions_col = st.columns(2)
        with reason_col:
            st.subheader("Why this employee may leave")
            for driver in selected["risk_drivers"]:
                st.markdown(f"- {driver}")
        with actions_col:
            st.subheader("Recommended HR actions")
            for action in selected["recommended_actions"]:
                st.markdown(f"- {action}")

        with st.expander("Why this reason was suggested"):
            st.plotly_chart(reason_distribution_chart(selected["reason_distribution"]), use_container_width=True)
            st.caption(
                "This reason is estimated from similar historical voluntary departures and current risk drivers. "
                "It should be treated as a hypothesis, not a certainty."
            )

            votes_frame = pd.DataFrame(
                {
                    "Historical raw reason": list(selected["raw_reason_votes"].keys()),
                    "Neighbor votes": list(selected["raw_reason_votes"].values()),
                }
            ).sort_values("Neighbor votes", ascending=False)
            if not votes_frame.empty:
                votes_chart = px.bar(
                    votes_frame,
                    x="Neighbor votes",
                    y="Historical raw reason",
                    orientation="h",
                    text="Neighbor votes",
                    color="Neighbor votes",
                    color_continuous_scale="Blues",
                )
                votes_chart.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                votes_chart.update_traces(textposition="outside")
                st.plotly_chart(votes_chart, use_container_width=True)

    with responsible_ai_tab:
        st.subheader("Responsible AI snapshot")
        st.markdown(
            """
            - **Explainable AI**: the dashboard relies on a Logistic Regression risk model so each employee can be explained.
            - **Ethical AI**: sensitive attributes are excluded from training and kept only for fairness audit.
            - **Model choice**: Logistic Regression is retained as the deployed model because it balances strong performance and clear HR-facing explanations better than the comparator model.
            - **Reason engine**: probable reasons are inferred from similar historical leavers and should be interpreted as decision-support hypotheses.
            """
        )

        fair_left, fair_right = st.columns(2)
        with fair_left:
            st.subheader("Fairness audit by sex")
            st.dataframe(sex_fairness, use_container_width=True, hide_index=True)
            st.plotly_chart(
                fairness_chart(sex_fairness, "Sex", "predicted_positive", "Predicted positive rate by sex"),
                use_container_width=True,
            )
        with fair_right:
            st.subheader("Fairness audit by race")
            st.dataframe(race_fairness, use_container_width=True, hide_index=True)
            st.plotly_chart(
                fairness_chart(
                    race_fairness,
                    "RaceDesc",
                    "predicted_positive",
                    "Predicted positive rate by race",
                ),
                use_container_width=True,
            )

        with st.expander("Model evaluation details"):
            st.dataframe(benchmark, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
