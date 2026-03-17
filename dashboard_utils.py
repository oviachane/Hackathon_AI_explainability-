from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "HRDataset_v14.csv"

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

MODEL_FEATURES = [
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "TenureYears",
    "SalaryVsPositionMedian",
    "AbsencesVsDeptMedian",
    "LateDaysVsDeptMedian",
    "PerformanceScore",
    "Position",
    "Department",
]

NUMERIC_FEATURES = [
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "TenureYears",
    "SalaryVsPositionMedian",
    "AbsencesVsDeptMedian",
    "LateDaysVsDeptMedian",
]

CATEGORICAL_FEATURES = [
    "PerformanceScore",
    "Position",
    "Department",
]

REASON_LABELS = {
    "Another position": "Other opportunity",
    "unhappy": "Dissatisfaction",
    "more money": "Compensation gap",
    "career change": "Career change",
    "hours": "Work hours or workload",
    "return to school": "Return to school",
    "relocation out of area": "Relocation",
    "maternity leave - did not return": "Post-maternity return issue",
    "retiring": "Retirement",
    "military": "Military constraints",
    "medical issues": "Health constraints",
}

REASON_ACTIONS = {
    "Another position": [
        "Open a mobility or role-evolution discussion immediately.",
        "Build a 6-month progression plan with concrete milestones.",
        "Assign a mentor or sponsor manager to secure the career path.",
    ],
    "unhappy": [
        "Launch a targeted retention interview with the manager and HR within 7 days.",
        "Review workload, team climate, and operational pain points around the role.",
        "Set a short engagement recovery plan with biweekly check-ins.",
    ],
    "more money": [
        "Compare compensation immediately against the internal role benchmark.",
        "Assess a targeted salary adjustment or retention package.",
        "Tie the pay discussion to a formal impact and growth plan.",
    ],
    "career change": [
        "Offer a reskilling path or internal role rotation.",
        "Identify a project assignment that rebuilds meaning and develops new skills.",
        "Schedule an HR career discussion to clarify credible internal options.",
    ],
    "hours": [
        "Review work-time organization and possible flexibility around the role.",
        "Analyze workload peaks, on-call pressure, and causes of operational fatigue.",
        "Test an adjusted schedule or hybrid setup if the role allows it.",
    ],
    "return to school": [
        "Explore part-time, study leave, or a temporary work arrangement.",
        "Assess co-funding or a training plan compatible with business needs.",
        "Keep the relationship open through a future return path if departure becomes inevitable.",
    ],
    "relocation out of area": [
        "Assess a site transfer or a hybrid work arrangement.",
        "Check whether the role can support partial or full remote work.",
        "Prepare a continuity plan if the relocation is already confirmed.",
    ],
    "maternity leave - did not return": [
        "Set up a flexible and gradual return-to-work path.",
        "Review schedule constraints and short-term managerial expectations.",
        "Formalize a dedicated HR follow-up on return and support needs.",
    ],
    "retiring": [
        "Set up phased retirement if relevant.",
        "Organize a knowledge-transfer and succession plan.",
        "Clarify stay-or-exit conditions with a realistic timeline.",
    ],
    "military": [
        "Document external obligations and possible scheduling flexibility.",
        "Prepare a continuity and reintegration path if applicable.",
        "Coordinate HR, planning, and management to reduce operational friction.",
    ],
    "medical issues": [
        "Offer role or schedule adjustments with occupational health support.",
        "Temporarily reduce workload or adapt the scope of responsibilities.",
        "Ensure confidential HR follow-up with regular check-ins.",
    ],
}

NON_ACTIONABLE_REASONS = {
    "return to school",
    "relocation out of area",
    "maternity leave - did not return",
    "retiring",
    "military",
    "medical issues",
}


def load_raw_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    parsed["DOB"] = pd.to_datetime(parsed["DOB"], format="%m/%d/%y", errors="coerce")

    for column in ["DateofHire", "DateofTermination", "LastPerformanceReview_Date"]:
        parsed[column] = pd.to_datetime(parsed[column], format="%m/%d/%Y", errors="coerce")

    return parsed


def _build_reference_date(df: pd.DataFrame) -> pd.Timestamp:
    candidates = []

    for column in ["LastPerformanceReview_Date", "DateofTermination", "DateofHire"]:
        valid_dates = df[column].dropna()
        if not valid_dates.empty:
            candidates.append(valid_dates.max())

    if not candidates:
        return pd.Timestamp("2019-12-31")

    return max(candidates).normalize()


def prepare_modeling_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = _parse_dates(raw_df)
    df = df[
        (df["TermReason"] == "N/A-StillEmployed")
        | (df["TermReason"].isin(VOLUNTARY_REASONS))
    ].copy()

    reference_date = _build_reference_date(df)

    df["TenureYears"] = (
        (reference_date - df["DateofHire"]).dt.days.div(365.25).clip(lower=0)
    )
    df["SalaryVsPositionMedian"] = (
        df["Salary"] - df.groupby("Position")["Salary"].transform("median")
    )
    df["AbsencesVsDeptMedian"] = (
        df["Absences"] - df.groupby("Department")["Absences"].transform("median")
    )
    df["LateDaysVsDeptMedian"] = (
        df["DaysLateLast30"] - df.groupby("Department")["DaysLateLast30"].transform("median")
    )

    df["EmployeeLabel"] = "EMP-" + df["EmpID"].astype(str)
    df = df.drop(columns=["Employee_Name"])

    return df, reference_date


def build_turnover_estimator() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=0.35,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_turnover_model(df: pd.DataFrame) -> dict[str, float]:
    X = df[MODEL_FEATURES]
    y = df["Termd"]
    estimator = build_turnover_estimator()

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        estimator,
        X,
        y,
        cv=splitter,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "accuracy": "accuracy",
            "f1": "f1",
        },
    )

    return {
        "roc_auc": float(np.mean(results["test_roc_auc"])),
        "average_precision": float(np.mean(results["test_average_precision"])),
        "accuracy": float(np.mean(results["test_accuracy"])),
        "f1": float(np.mean(results["test_f1"])),
    }


def _compute_signal_profile(active_df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(index=active_df.index)

    profile["low_engagement"] = 1 - active_df["EngagementSurvey"].rank(pct=True)
    profile["low_satisfaction"] = 1 - active_df["EmpSatisfaction"].rank(pct=True)
    profile["high_absences"] = active_df["Absences"].rank(pct=True)
    profile["high_lateness"] = active_df["DaysLateLast30"].rank(pct=True)
    profile["underpaid"] = 1 - active_df["SalaryVsPositionMedian"].rank(pct=True)
    profile["long_tenure"] = active_df["TenureYears"].rank(pct=True)
    profile["low_projects"] = 1 - active_df["SpecialProjectsCount"].rank(pct=True)
    profile["high_performance"] = active_df["PerformanceScore"].map(
        {
            "Exceeds": 1.0,
            "Fully Meets": 0.8,
            "Needs Improvement": 0.4,
            "PIP": 0.2,
        }
    ).fillna(0.5)

    profile["signal_score"] = (
        profile["low_engagement"] * 0.22
        + profile["low_satisfaction"] * 0.22
        + profile["high_absences"] * 0.18
        + profile["high_lateness"] * 0.12
        + profile["underpaid"] * 0.16
        + profile["long_tenure"] * 0.10
    )

    return profile


def _weighted_reason_vote(
    active_df: pd.DataFrame, leavers_df: pd.DataFrame, transformed_active, transformed_leavers
) -> pd.DataFrame:
    neighbor_count = min(5, len(leavers_df))
    matcher = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    matcher.fit(transformed_leavers)

    distances, indices = matcher.kneighbors(transformed_active)
    reason_rows = []

    for row_index, row_distances, row_indices in zip(active_df.index, distances, indices):
        weights: dict[str, float] = {}

        for distance, neighbor_index in zip(row_distances, row_indices):
            reason = leavers_df.iloc[neighbor_index]["TermReason"]
            weights[reason] = weights.get(reason, 0.0) + (1.0 / (distance + 0.25))

        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        top_reason, top_weight = ranked[0]
        confidence = top_weight / sum(weight for _, weight in ranked)

        reason_rows.append(
            {
                "index": row_index,
                "neighbor_reason": top_reason,
                "neighbor_confidence": confidence,
            }
        )

    return pd.DataFrame(reason_rows).set_index("index")


def _theme_scores(row: pd.Series) -> dict[str, float]:
    return {
        "unhappy": (
            row["low_satisfaction"] * 0.35
            + row["low_engagement"] * 0.30
            + row["high_absences"] * 0.20
            + row["high_lateness"] * 0.15
        ),
        "more money": (
            row["underpaid"] * 0.40
            + row["high_performance"] * 0.20
            + row["long_tenure"] * 0.20
            + row["low_satisfaction"] * 0.20
        ),
        "Another position": (
            row["high_performance"] * 0.30
            + row["long_tenure"] * 0.25
            + row["low_projects"] * 0.25
            + row["underpaid"] * 0.20
        ),
        "career change": (
            row["low_projects"] * 0.30
            + row["long_tenure"] * 0.25
            + row["low_engagement"] * 0.20
            + row["high_performance"] * 0.25
        ),
        "hours": (
            row["high_lateness"] * 0.35
            + row["high_absences"] * 0.30
            + row["low_satisfaction"] * 0.20
            + row["underpaid"] * 0.15
        ),
    }


def _select_reason(row: pd.Series) -> tuple[str, str]:
    scored_themes = _theme_scores(row)
    fallback_reason = max(scored_themes, key=scored_themes.get)

    if row["neighbor_confidence"] >= 0.60:
        selected_reason = row["neighbor_reason"]
        reason_source = (
            f"Reason suggested by strong historical similarity ({row['neighbor_confidence']:.0%})."
        )
    elif (
        row["neighbor_confidence"] >= 0.38
        and row["neighbor_reason"] not in NON_ACTIONABLE_REASONS
    ):
        selected_reason = row["neighbor_reason"]
        reason_source = (
            f"Reason suggested by historical similarity ({row['neighbor_confidence']:.0%})."
        )
    else:
        selected_reason = fallback_reason
        reason_source = "Reason inferred from individual HR signals."

    return selected_reason, reason_source


def _format_currency(value: float) -> str:
    return f"{value:,.0f} $".replace(",", " ")


def _build_warning_signals(row: pd.Series) -> list[str]:
    signals = []

    if row["low_satisfaction"] >= 0.55 or row["EmpSatisfaction"] <= 2:
        signals.append(
            (row["low_satisfaction"], f"Low satisfaction ({int(row['EmpSatisfaction'])}/5).")
        )

    if row["low_engagement"] >= 0.55 or row["EngagementSurvey"] <= 3.25:
        signals.append(
            (
                row["low_engagement"],
                f"Low engagement ({row['EngagementSurvey']:.2f}/5).",
            )
        )

    if row["Absences"] >= 10 or row["high_absences"] >= 0.70:
        signals.append(
            (
                row["high_absences"],
                f"High absence level ({int(row['Absences'])} days).",
            )
        )

    if row["DaysLateLast30"] > 0 and row["high_lateness"] >= 0.70:
        signals.append(
            (
                row["high_lateness"],
                f"Significant recent lateness ({int(row['DaysLateLast30'])} days in the last 30).",
            )
        )

    if row["SalaryVsPositionMedian"] < 0 and row["underpaid"] >= 0.55:
        signals.append(
            (
                row["underpaid"],
                "Compensation below the role median "
                f"({ _format_currency(row['SalaryVsPositionMedian']) }).",
            )
        )

    if row["long_tenure"] >= 0.70:
        signals.append(
            (
                row["long_tenure"],
                f"High tenure ({row['TenureYears']:.1f} years), with a possible stagnation risk.",
            )
        )

    if row["SpecialProjectsCount"] <= 1 and row["low_projects"] >= 0.60:
        signals.append(
            (
                row["low_projects"],
                f"Low exposure to special projects ({int(row['SpecialProjectsCount'])}).",
            )
        )

    if not signals:
        signals.append(
            (
                row["signal_score"],
                "Combined moderate signals across role context, tenure, and HR history.",
            )
        )

    signals = sorted(signals, key=lambda item: item[0], reverse=True)
    return [message for _, message in signals[:3]]


def _risk_band(score: float) -> str:
    if score >= 0.80:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.50:
        return "Moderate"
    return "Monitor"


def build_dashboard_dataset(data_path: Path = DATA_PATH) -> dict[str, object]:
    raw_df = load_raw_dataset(data_path)
    model_df, reference_date = prepare_modeling_dataset(raw_df)
    metrics = evaluate_turnover_model(model_df)

    estimator = build_turnover_estimator()
    estimator.fit(model_df[MODEL_FEATURES], model_df["Termd"])

    active_df = model_df[model_df["Termd"] == 0].copy()
    active_df = pd.concat([active_df, _compute_signal_profile(active_df)], axis=1)
    active_df["model_risk"] = estimator.predict_proba(active_df[MODEL_FEATURES])[:, 1]
    active_df["final_risk"] = (
        active_df["model_risk"] * 0.70 + active_df["signal_score"] * 0.30
    )

    leavers_df = model_df[model_df["Termd"] == 1].copy()
    preprocessor = estimator.named_steps["preprocessor"]
    active_transformed = preprocessor.transform(active_df[MODEL_FEATURES])
    leavers_transformed = preprocessor.transform(leavers_df[MODEL_FEATURES])

    active_df = active_df.join(
        _weighted_reason_vote(
            active_df=active_df,
            leavers_df=leavers_df,
            transformed_active=active_transformed,
            transformed_leavers=leavers_transformed,
        )
    )

    enrichments = []

    for index, row in active_df.iterrows():
        selected_reason, reason_source = _select_reason(row)
        warning_signals = _build_warning_signals(row)

        enrichments.append(
            {
                "index": index,
                "selected_reason": selected_reason,
                "reason_label": REASON_LABELS[selected_reason],
                "reason_source": reason_source,
                "warning_signals": warning_signals,
                "hr_actions": REASON_ACTIONS[selected_reason],
                "risk_band": _risk_band(row["final_risk"]),
            }
        )

    active_df = active_df.join(pd.DataFrame(enrichments).set_index("index"))
    active_df["model_risk_pct"] = (active_df["model_risk"] * 100).round(1)
    active_df["final_risk_pct"] = (active_df["final_risk"] * 100).round(1)
    active_df["neighbor_confidence_pct"] = (active_df["neighbor_confidence"] * 100).round(1)
    active_df = active_df.sort_values(
        ["final_risk", "model_risk", "signal_score"],
        ascending=False,
    ).reset_index(drop=True)
    active_df["priority_rank"] = active_df.index + 1

    top_five = active_df.head(5).copy()

    department_summary = (
        active_df.groupby("Department", dropna=False)
        .agg(
            active_employees=("EmpID", "size"),
            avg_final_risk=("final_risk", "mean"),
            high_risk_employees=("final_risk", lambda values: int((values >= 0.65).sum())),
        )
        .reset_index()
        .sort_values("avg_final_risk", ascending=False)
    )
    department_summary["avg_final_risk_pct"] = (
        department_summary["avg_final_risk"] * 100
    ).round(1)

    risky_population = active_df[active_df["final_risk"] >= 0.65].copy()
    if risky_population.empty:
        risky_population = active_df.head(15).copy()

    reason_summary = (
        risky_population.groupby("reason_label")
        .size()
        .reset_index(name="employees")
        .sort_values("employees", ascending=False)
    )

    export_columns = [
        "priority_rank",
        "EmployeeLabel",
        "Position",
        "Department",
        "final_risk_pct",
        "model_risk_pct",
        "reason_label",
        "reason_source",
    ]
    export_df = top_five[export_columns].copy()
    export_df["warning_signals"] = top_five["warning_signals"].apply(" | ".join)
    export_df["hr_actions"] = top_five["hr_actions"].apply(" | ".join)

    return {
        "reference_date": reference_date,
        "metrics": metrics,
        "model_df": model_df,
        "active_df": active_df,
        "top_five": top_five,
        "department_summary": department_summary,
        "reason_summary": reason_summary,
        "export_df": export_df,
    }
