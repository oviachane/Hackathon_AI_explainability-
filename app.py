from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard_utils import build_dashboard_dataset


TEXT_COLOR = "#1d2a38"
MUTED_TEXT_COLOR = "#4a5565"
ACCENT_COLOR = "#133c55"
CARD_BG = "rgba(255, 255, 255, 0.88)"


st.set_page_config(
    page_title="HR Dashboard - Voluntary Turnover",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_dashboard():
    return build_dashboard_dataset()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(250, 210, 168, 0.36), transparent 35%),
                radial-gradient(circle at top right, rgba(129, 178, 154, 0.18), transparent 28%),
                linear-gradient(180deg, #f6f1e8 0%, #eef2ef 48%, #f8faf8 100%);
            color: #1d2a38;
            font-family: Georgia, "Iowan Old Style", serif;
        }
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stCaptionContainer"] {
            color: #1d2a38;
        }
        .stSidebar {
            background: linear-gradient(180deg, rgba(255,255,255,0.90), rgba(244, 240, 232, 0.94));
        }
        .stSidebar,
        .stSidebar p,
        .stSidebar li,
        .stSidebar label,
        .stSidebar [data-testid="stMarkdownContainer"] {
            color: #1d2a38;
        }
        .stSelectbox > div[data-baseweb="select"] > div,
        .stMultiSelect > div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.96);
            color: #1d2a38;
        }
        .stCheckbox label,
        .stSlider label {
            color: #1d2a38;
        }
        .stDownloadButton button,
        .stButton button {
            background: #133c55;
            color: #f9f6f0;
            border: 1px solid rgba(19, 60, 85, 0.15);
        }
        .stDownloadButton button:hover,
        .stButton button:hover {
            background: #245c73;
            color: #ffffff;
        }
        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.90);
            border-radius: 18px;
            padding: 0.2rem;
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #133c55 0%, #245c73 58%, #e07a5f 100%);
            color: #f9f6f0;
            box-shadow: 0 18px 45px rgba(19, 60, 85, 0.20);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.25rem;
            margin: 0 0 0.4rem 0;
            font-weight: 700;
        }
        .hero p {
            margin: 0.15rem 0;
            font-size: 1rem;
            opacity: 0.95;
        }
        .note-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(19, 60, 85, 0.10);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(17, 24, 39, 0.06);
            color: #1d2a38;
        }
        .employee-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(19, 60, 85, 0.10);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 12px 26px rgba(17, 24, 39, 0.06);
            margin-bottom: 0.85rem;
        }
        .employee-title {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.35rem;
        }
        .employee-title h3 {
            margin: 0;
            font-size: 1.15rem;
            color: #133c55;
        }
        .risk-pill {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            background: #f7d8cf;
            color: #8d2d19;
        }
        .employee-meta {
            font-size: 0.92rem;
            color: #4a5565;
            margin-bottom: 0.7rem;
        }
        .small-label {
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5c6c7f;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(19, 60, 85, 0.10);
            padding: 0.9rem 1rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(17, 24, 39, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure):
    figure.update_layout(
        font=dict(color=TEXT_COLOR, family='Georgia, "Iowan Old Style", serif'),
        title_font=dict(color=ACCENT_COLOR, size=20),
        xaxis=dict(
            color=TEXT_COLOR,
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(19, 60, 85, 0.12)",
            zerolinecolor="rgba(19, 60, 85, 0.12)",
        ),
        yaxis=dict(
            color=TEXT_COLOR,
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
            gridcolor="rgba(19, 60, 85, 0.12)",
            zerolinecolor="rgba(19, 60, 85, 0.12)",
        ),
        legend=dict(font=dict(color=TEXT_COLOR)),
        coloraxis_colorbar=dict(
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            font=dict(color=TEXT_COLOR),
            bordercolor="rgba(19, 60, 85, 0.16)",
        ),
        plot_bgcolor="rgba(255,255,255,0.28)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def display_name(row) -> str:
    return row["EmployeeLabel"]


def render_employee_card(row) -> None:
    name = display_name(row)

    st.markdown(
        f"""
        <div class="employee-card">
            <div class="employee-title">
                <h3>#{int(row['priority_rank'])} · {name}</h3>
                <span class="risk-pill">{row['risk_band']} · {row['final_risk_pct']:.1f}%</span>
            </div>
            <div class="employee-meta">
                {row['Position']} · {row['Department']} · tenure {row['TenureYears']:.1f} years
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown('<div class="small-label">Why this profile is flagged</div>', unsafe_allow_html=True)
        for signal in row["warning_signals"]:
            st.markdown(f"- {signal}")

    with right:
        st.markdown('<div class="small-label">Likely reason and HR action</div>', unsafe_allow_html=True)
        st.markdown(f"**Likely reason**: {row['reason_label']}")
        st.caption(row["reason_source"])
        for action in row["hr_actions"]:
            st.markdown(f"- {action}")

    st.caption(
        "Priority score: "
        f"{row['final_risk_pct']:.1f}% | "
        f"model score {row['model_risk_pct']:.1f}% | "
        f"similarity confidence {row['neighbor_confidence_pct']:.1f}%"
    )


inject_styles()
dashboard = load_dashboard()

active_df = dashboard["active_df"].copy()
department_summary = dashboard["department_summary"].copy()
reason_summary = dashboard["reason_summary"].copy()
metrics = dashboard["metrics"]
reference_date = dashboard["reference_date"]

with st.sidebar:
    st.markdown("## Settings")
    selected_department = st.selectbox(
        "Department filter",
        ["All"] + sorted(active_df["Department"].dropna().unique().tolist()),
    )
    top_n = st.slider("Number of priority profiles", min_value=5, max_value=12, value=5)

    st.markdown("---")
    st.markdown("## Guardrails")
    st.markdown("- Involuntary exits are excluded from scope.")
    st.markdown("- Sensitive attributes are excluded from scoring.")
    st.markdown("- Logistic Regression is used to keep the model transparent.")
    st.markdown("- Employee names are permanently hidden.")

filtered_df = active_df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_department].copy()

filtered_df = filtered_df.sort_values(
    ["final_risk", "model_risk", "signal_score"],
    ascending=False,
).reset_index(drop=True)
filtered_df["priority_rank"] = filtered_df.index + 1

shortlist = filtered_df.head(top_n).copy()

if shortlist.empty:
    st.error("No employee matches the selected filter.")
    st.stop()

table_view = shortlist.assign(
    Employee=shortlist.apply(display_name, axis=1),
    Risk=shortlist["final_risk_pct"].map(lambda value: f"{value:.1f}%"),
    LikelyReason=shortlist["reason_label"],
    RiskLevel=shortlist["risk_band"],
)[
    [
        "priority_rank",
        "Employee",
        "Position",
        "Department",
        "Risk",
        "LikelyReason",
        "RiskLevel",
    ]
].rename(
    columns={
        "priority_rank": "Rank",
        "Position": "Role",
        "Department": "Department",
        "LikelyReason": "Likely reason",
        "RiskLevel": "Risk level",
    }
)

export_df = shortlist[
    [
        "priority_rank",
        "EmployeeLabel",
        "Position",
        "Department",
        "final_risk_pct",
        "model_risk_pct",
        "reason_label",
        "reason_source",
        "warning_signals",
        "hr_actions",
    ]
].copy()
export_df["warning_signals"] = export_df["warning_signals"].apply(" | ".join)
export_df["hr_actions"] = export_df["hr_actions"].apply(" | ".join)

st.markdown(
    f"""
    <div class="hero">
        <h1>HR Dashboard · Voluntary Turnover</h1>
        <p>Top at-risk profiles, likely resignation drivers, and retention actions for HR.</p>
        <p>Analytical reference date: {reference_date.strftime('%d/%m/%Y')} · Priority score = 70% explainable model + 30% actionable HR signals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Active employees monitored", f"{len(active_df)}")
metric_col2.metric(
    "Voluntary exits used for training",
    f"{int((dashboard['model_df']['Termd'] == 1).sum())}",
)
metric_col3.metric("ROC-AUC CV", f"{metrics['roc_auc']:.2f}")
metric_col4.metric("Average top 5 risk", f"{dashboard['top_five']['final_risk_pct'].mean():.1f}%")

table_col, note_col = st.columns([1.55, 0.95])

with table_col:
    st.subheader("Priority shortlist")
    styled_table = table_view.style.set_properties(
        **{"background-color": CARD_BG, "color": TEXT_COLOR}
    )
    styled_table = styled_table.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#133c55"),
                    ("color", "#f9f6f0"),
                    ("font-weight", "700"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid rgba(19, 60, 85, 0.08)"),
                ],
            },
        ]
    )
    st.dataframe(styled_table, hide_index=True, use_container_width=True)
    st.download_button(
        "Download priority shortlist (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="top_turnover_priority.csv",
        mime="text/csv",
    )

with note_col:
    st.markdown(
        """
        <div class="note-card">
            <div class="small-label">Business reading</div>
            <p style="margin: 0.2rem 0 0.7rem 0;">
                The priority score is not a judgment on the employee. It ranks the cases
                where HR can intervene quickly with an explainable rationale.
            </p>
            <div class="small-label">What this dashboard delivers</div>
            <p style="margin: 0.2rem 0;">
                1. A shortlist of the most exposed profiles.
                <br>2. A likely resignation reason.
                <br>3. Concrete HR actions for each case.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

chart_col1, chart_col2 = st.columns(2)

department_fig = px.bar(
    department_summary.head(8).sort_values("avg_final_risk_pct"),
    x="avg_final_risk_pct",
    y="Department",
    orientation="h",
    color="high_risk_employees",
    color_continuous_scale=["#81b29a", "#f2cc8f", "#e07a5f"],
    text="avg_final_risk_pct",
)
department_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
department_fig.update_layout(
    title="Most exposed departments",
    xaxis_title="Average resignation risk",
    yaxis_title="",
    coloraxis_colorbar_title="High-risk cases",
    margin=dict(l=10, r=10, t=50, b=10),
)
department_fig.update_traces(textfont=dict(color=TEXT_COLOR))
style_figure(department_fig)

reason_fig = px.bar(
    reason_summary.sort_values("employees"),
    x="employees",
    y="reason_label",
    orientation="h",
    color="employees",
    color_continuous_scale=["#81b29a", "#f2cc8f", "#e07a5f"],
    text="employees",
)
reason_fig.update_layout(
    title="Likely reasons among already high-risk profiles",
    xaxis_title="Number of employees",
    yaxis_title="",
    coloraxis_showscale=False,
    margin=dict(l=10, r=10, t=50, b=10),
)
reason_fig.update_traces(textfont=dict(color=TEXT_COLOR))
style_figure(reason_fig)

with chart_col1:
    st.plotly_chart(department_fig, use_container_width=True)

with chart_col2:
    st.plotly_chart(reason_fig, use_container_width=True)

scatter_df = filtered_df.assign(
    DisplayName=filtered_df.apply(display_name, axis=1)
)

scatter_fig = px.scatter(
    scatter_df,
    x="signal_score",
    y="model_risk",
    size="Absences",
    color="final_risk",
    hover_name="DisplayName",
    hover_data={
        "Department": True,
        "Position": True,
        "Absences": True,
        "DaysLateLast30": True,
        "signal_score": ":.2f",
        "model_risk": ":.2f",
        "final_risk": ":.2f",
    },
    color_continuous_scale=["#81b29a", "#f2cc8f", "#e07a5f"],
)
scatter_fig.update_layout(
    title="Model probability vs HR signals",
    xaxis_title="HR signal score",
    yaxis_title="Model probability",
    margin=dict(l=10, r=10, t=50, b=10),
)
style_figure(scatter_fig)
st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("Detailed priority profiles")
for _, employee in shortlist.iterrows():
    render_employee_card(employee)
