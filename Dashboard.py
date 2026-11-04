from pathlib import Path
import html

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = go = None
    PLOTLY_AVAILABLE = False


st.set_page_config(
    page_title="Breast Cancer | Patient Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14243A"
NAVY_2 = "#203650"
PINK = "#D87598"
PINK_LIGHT = "#F6E7ED"
GRAY = "#687789"
PALE = "#F4F6F9"
WHITE = "#FFFFFF"
PALETTE = [PINK, NAVY_2, "#83A7B8", "#D8B56D", "#A99BC4", "#93B79F"]


def add_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root {{ --navy:{NAVY}; --pink:{PINK}; --muted:{GRAY}; }}
        html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
        .stApp {{ background:{PALE}; color:{NAVY}; }}
        [data-testid="stSidebar"] {{ background:{NAVY}; }}
        [data-testid="stSidebar"] * {{ color:#F5F7FB; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:8px 10px; border-radius:9px; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:#263B55; }}
        .block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1500px; }}
        h1,h2,h3 {{ font-family:'Manrope',sans-serif; color:{NAVY}; letter-spacing:-.025em; }}
        h1 {{ font-size:2.15rem; font-weight:800; }}
        h2 {{ font-size:1.25rem; font-weight:700; }}
        [data-testid="stMetric"] {{ background:{WHITE}; padding:19px 20px; border:1px solid #E8ECF1;
            border-radius:16px; box-shadow:0 5px 18px rgba(20,36,58,.055); min-height:118px; }}
        [data-testid="stMetricLabel"] {{ color:{GRAY}; font-size:.85rem; }}
        [data-testid="stMetricValue"] {{ color:{NAVY}; font-family:'Manrope',sans-serif; font-weight:800; }}
        .hero {{ background:linear-gradient(115deg,{NAVY} 0%,#263F5B 72%,#755064 130%); color:white;
            padding:27px 31px; border-radius:20px; margin:0 0 24px; box-shadow:0 12px 28px rgba(20,36,58,.14); }}
        .hero h1 {{ color:white; margin:0 0 6px; }}
        .hero p {{ color:#DCE5EE; margin:0; font-size:.98rem; }}
        .eyebrow {{ color:#F0B5C9; font-size:.74rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; margin-bottom:8px; }}
        .section-note {{ color:{GRAY}; margin-top:-10px; margin-bottom:20px; }}
        .insight {{ background:white; border:1px solid #E8ECF1; border-left:4px solid {PINK};
            padding:16px 18px; border-radius:12px; margin:9px 0; box-shadow:0 4px 14px rgba(20,36,58,.04); }}
        .insight strong {{ color:{NAVY}; }} .insight span {{ color:{GRAY}; }}
        .footnote {{ color:{GRAY}; font-size:.78rem; padding-top:12px; }}
        div[data-testid="stPlotlyChart"], div[data-testid="stImage"] {{ background:white; border-radius:15px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading patient records…")
def load_data():
    csv_path = Path(__file__).resolve().parent / "Breast_cancer.csv" / "Breast_Cancer.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found beside the dashboard: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    # Match the original EDA: remove exact duplicate records, then normalize text whitespace.
    df = df.drop_duplicates().copy()
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column].dtype):
            df[column] = df[column].str.strip()
    numeric_columns = ["age", "tumor_size", "regional_node_examined", "reginol_node_positive", "survival_months", "grade"]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "tumor_size" in df:
        df["tumor_size_category"] = pd.cut(
            df["tumor_size"], bins=[0, 20, 50, float("inf")],
            labels=["Small", "Medium", "Large"], include_lowest=True,
        )
    if "age" in df:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 39, 49, 59, 69, float("inf")],
            labels=["<40", "40–49", "50–59", "60–69", "70+"], include_lowest=True,
        )
    return df


def fmt_number(value, digits=1):
    return "—" if pd.isna(value) else f"{value:,.{digits}f}"


def plot_chart(fig, fallback_data=None, fallback_kind="bar", x=None, y=None, color=None, title=None):
    if PLOTLY_AVAILABLE:
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(family="DM Sans", color=NAVY),
            margin=dict(l=18, r=18, t=48, b=18), title=dict(text=title or "", font=dict(size=15)),
            legend_title_text="", colorway=PALETTE,
        )
        fig.update_xaxes(showgrid=False, linecolor="#E8ECF1")
        fig.update_yaxes(gridcolor="#EDF0F4", zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    elif fallback_data is not None:
        st.caption(title or "")
        if fallback_kind == "scatter":
            st.scatter_chart(fallback_data, x=x, y=y, color=color, width="stretch")
        elif fallback_kind == "line":
            st.line_chart(fallback_data, x=x, y=y, color=color, width="stretch")
        else:
            st.bar_chart(fallback_data, x=x, y=y, color=color, width="stretch")
    else:
        st.info("Install Plotly (`pip install plotly`) to display this interactive chart.")


def categorical_chart(df, column, title, color=None, normalize=False, horizontal=False):
    data = df.dropna(subset=[column]).copy()
    if PLOTLY_AVAILABLE:
        if color:
            grouped = data.groupby([column, color], observed=False).size().reset_index(name="Patients")
            if normalize:
                grouped["Patients"] = grouped["Patients"] / grouped.groupby(column)["Patients"].transform("sum") * 100
            fig = px.bar(grouped, x="Patients" if horizontal else column,
                         y=column if horizontal else "Patients", color=color, barmode="stack",
                         orientation="h" if horizontal else "v", title=title)
        else:
            counts = data[column].value_counts(dropna=True).rename_axis(column).reset_index(name="Patients")
            fig = px.bar(counts, x="Patients" if horizontal else column,
                         y=column if horizontal else "Patients", orientation="h" if horizontal else "v", title=title)
        if normalize:
            fig.update_layout(yaxis_title="Share of group (%)", xaxis_title="" if not horizontal else "Share of group (%)")
            fig.update_yaxes(tickformat=".0f" if horizontal else ".0%")
        plot_chart(fig)
    else:
        if color:
            counts = pd.crosstab(data[column], data[color], normalize="index" if normalize else False)
            if normalize:
                counts = counts * 100
            st.caption(title)
            st.bar_chart(counts, width="stretch")
        else:
            counts = data[column].value_counts().rename_axis(column).reset_index(name="Patients")
            plot_chart(None, counts, x=column, y="Patients", title=title)


def numeric_histogram(df, column, title, color=None, bins=25):
    data = df.dropna(subset=[column])
    if PLOTLY_AVAILABLE:
        fig = px.histogram(data, x=column, color=color, nbins=bins, barmode="overlay", opacity=.82, title=title)
        fig.update_traces(marker_line_width=0)
        plot_chart(fig)
    else:
        plot_chart(None, data, fallback_kind="bar", x=column, title=title)


def status_means(df, measure):
    return df.groupby("status", dropna=True)[measure].mean().sort_index().rename("Average").reset_index()


def heading(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="section-note">{subtitle}</div>', unsafe_allow_html=True)


def render_overview(df, all_df):
    heading("Overview", "A concise view of the filtered patient cohort.")
    total = len(df)
    alive = int((df["status"] == "Alive").sum())
    dead = int((df["status"] == "Dead").sum())
    cols = st.columns(5)
    metrics = [
        ("Patients", f"{total:,}"), ("Average age", f"{fmt_number(df.age.mean())} yrs"),
        ("Average tumor size", fmt_number(df.tumor_size.mean()) + " mm"),
        ("Alive", f"{alive:,}"), ("Deceased", f"{dead:,}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    if total:
        st.markdown(f'<div class="footnote">Alive share in the current selection: <b>{alive/total:.1%}</b> · Mean recorded survival: <b>{fmt_number(df.survival_months.mean())} months</b>. These are descriptive dataset measures.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        categorical_chart(df, "status", "Patient status")
    with right:
        categorical_chart(df, "age_group", "Patients by age group")
    left, right = st.columns(2)
    with left:
        if PLOTLY_AVAILABLE:
            means = status_means(df, "survival_months")
            fig = px.bar(means, x="status", y="Average", title="Mean recorded survival by status", text_auto=".1f")
            fig.update_yaxes(title="Months")
            plot_chart(fig)
        else:
            means = status_means(df, "survival_months")
            plot_chart(None, means, x="status", y="Average", title="Mean recorded survival by status")
    with right:
        categorical_chart(df, "6th_stage", "Cancer stage mix")


def render_demographics(df):
    heading("Patient demographics", "Explore the age profile and recorded social demographics.")
    a, b = st.columns(2)
    with a:
        numeric_histogram(df, "age", "Age distribution", bins=20)
    with b:
        categorical_chart(df, "age_group", "Age group")
    a, b = st.columns(2)
    with a:
        categorical_chart(df, "race", "Race distribution")
    with b:
        categorical_chart(df, "marital_status", "Marital status")


def render_characteristics(df):
    heading("Cancer characteristics", "Tumor burden, stage, receptor status, and lymph node findings.")
    a, b = st.columns(2)
    with a:
        numeric_histogram(df, "tumor_size", "Tumor size distribution", bins=28)
    with b:
        categorical_chart(df, "tumor_size_category", "Tumor size category by status", color="status")
    a, b = st.columns(2)
    with a:
        categorical_chart(df, "t_stage", "T Stage")
    with b:
        categorical_chart(df, "n_stage", "N Stage")
    a, b = st.columns(2)
    with a:
        categorical_chart(df, "6th_stage", "Cancer stage by status", color="status")
    with b:
        categorical_chart(df, "a_stage", "Distant metastasis stage")
    a, b = st.columns(2)
    with a:
        categorical_chart(df, "estrogen_status", "Estrogen receptor status")
        categorical_chart(df, "progesterone_status", "Progesterone receptor status")
    with b:
        if PLOTLY_AVAILABLE:
            fig = px.scatter(df, x="regional_node_examined", y="reginol_node_positive", color="status",
                             hover_data=["age", "6th_stage"], title="Nodes examined and positive")
            plot_chart(fig)
        else:
            plot_chart(None, df, fallback_kind="scatter", x="regional_node_examined", y="reginol_node_positive", color="status", title="Nodes examined and positive")
        st.caption(f"Mean examined: {fmt_number(df.regional_node_examined.mean())} · Mean positive: {fmt_number(df.reginol_node_positive.mean())}")


def render_survival(df):
    heading("Survival analysis", "Recorded survival months and descriptive comparisons across patient groups.")
    a, b = st.columns(2)
    with a:
        numeric_histogram(df, "survival_months", "Recorded survival months", color="status", bins=28)
    with b:
        if PLOTLY_AVAILABLE:
            fig = px.box(df, x="status", y="survival_months", color="status", points="outliers", title="Survival months by status")
            plot_chart(fig)
        else:
            summary = df.groupby("status")["survival_months"].describe()[["min", "25%", "50%", "75%", "max"]]
            st.caption("Survival months by status (five-number summary)")
            st.dataframe(summary, width="stretch")
    a, b = st.columns(2)
    with a:
        if PLOTLY_AVAILABLE:
            grouped = df.groupby(["age_group", "status"], observed=False)["survival_months"].mean().reset_index()
            fig = px.bar(grouped, x="age_group", y="survival_months", color="status", barmode="group", title="Mean recorded survival by age group")
            fig.update_yaxes(title="Mean months")
            plot_chart(fig)
        else:
            grouped = df.groupby(["age_group", "status"], observed=False)["survival_months"].mean().unstack()
            st.caption("Mean recorded survival by age group")
            st.bar_chart(grouped, width="stretch")
    with b:
        if PLOTLY_AVAILABLE:
            fig = px.scatter(df, x="tumor_size", y="survival_months", color="status",
                             hover_data=["age", "6th_stage"], opacity=.72, title="Tumor size and recorded survival")
            plot_chart(fig)
        else:
            plot_chart(None, df, fallback_kind="scatter", x="tumor_size", y="survival_months", color="status", title="Tumor size and recorded survival")
    if PLOTLY_AVAILABLE:
        fig = px.scatter(df, x="age", y="survival_months", color="status", hover_data=["tumor_size", "6th_stage"],
                         title="Age and recorded survival", opacity=.7)
        plot_chart(fig)


def render_insights(all_df, df):
    heading("Data insights", "Findings calculated directly from the deduplicated source records.")
    overall = all_df
    status_pct = overall["status"].value_counts(normalize=True) * 100
    status_age = overall.groupby("status")["age"].mean()
    status_tumor = overall.groupby("status")["tumor_size"].mean()
    status_survival = overall.groupby("status")["survival_months"].mean()
    size_status = pd.crosstab(overall["tumor_size_category"], overall["status"], normalize="index") * 100
    if "Alive" in status_pct and "Dead" in status_pct:
        st.markdown(f'<div class="insight"><strong>Status mix</strong><br><span>{status_pct["Alive"]:.2f}% of deduplicated records are marked Alive and {status_pct["Dead"]:.2f}% Dead.</span></div>', unsafe_allow_html=True)
    if "Alive" in status_age and "Dead" in status_age:
        st.markdown(f'<div class="insight"><strong>Age by recorded status</strong><br><span>Mean age: {status_age["Alive"]:.2f} years for Alive records and {status_age["Dead"]:.2f} years for Dead records.</span></div>', unsafe_allow_html=True)
    if "Alive" in status_tumor and "Dead" in status_tumor:
        st.markdown(f'<div class="insight"><strong>Tumor size by recorded status</strong><br><span>Mean tumor size: {status_tumor["Alive"]:.2f} for Alive records and {status_tumor["Dead"]:.2f} for Dead records.</span></div>', unsafe_allow_html=True)
    if "Alive" in status_survival and "Dead" in status_survival:
        st.markdown(f'<div class="insight"><strong>Recorded survival months</strong><br><span>Mean recorded duration: {status_survival["Alive"]:.2f} months in Alive records and {status_survival["Dead"]:.2f} months in Dead records.</span></div>', unsafe_allow_html=True)
    if not size_status.empty and "Alive" in size_status and "Dead" in size_status:
        small_alive = size_status.loc["Small", "Alive"] if "Small" in size_status.index else float("nan")
        large_dead = size_status.loc["Large", "Dead"] if "Large" in size_status.index else float("nan")
        st.markdown(f'<div class="insight"><strong>Tumor size category</strong><br><span>Alive share of Small category: {small_alive:.2f}%; Dead share of Large category: {large_dead:.2f}%.</span></div>', unsafe_allow_html=True)
    st.markdown("<div class='footnote'>These are descriptive associations in this dataset. Status and recorded survival duration do not establish treatment effects, prognosis, or causation.</div>", unsafe_allow_html=True)
    if len(df) != len(overall):
        st.caption(f"The sidebar filters currently select {len(df):,} of {len(overall):,} records; insight cards above intentionally report full dataset calculations.")


def main():
    add_styles()
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
    required = {"age", "tumor_size", "survival_months", "status", "race", "marital_status", "6th_stage"}
    missing = sorted(required - set(data.columns))
    if missing:
        st.error("The CSV is missing required columns: " + ", ".join(missing))
        st.stop()

    with st.sidebar:
        st.markdown("<div class='eyebrow'>Clinical analytics</div><h2 style='color:white'>Breast Cancer<br>Patient Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#3A4D64'>", unsafe_allow_html=True)
        section = st.radio("Navigate", ["Overview", "Patient demographics", "Cancer characteristics", "Survival analysis", "Data insights"], label_visibility="collapsed")
        st.markdown("<hr style='border-color:#3A4D64'>", unsafe_allow_html=True)
        st.markdown("#### Cohort filters")
        ages = sorted(data["age_group"].dropna().astype(str).unique().tolist())
        selected_ages = st.multiselect("Age group", ages, default=ages)
        races = sorted(data["race"].dropna().unique().tolist())
        selected_races = st.multiselect("Race", races, default=races)
        stages = sorted(data["6th_stage"].dropna().unique().tolist())
        selected_stages = st.multiselect("Cancer stage", stages, default=stages)
        statuses = sorted(data["status"].dropna().unique().tolist())
        selected_statuses = st.multiselect("Patient status", statuses, default=statuses)
        st.markdown("<div class='footnote'>Source: Breast_cancer.csv<br>Exact duplicates removed; text whitespace trimmed.</div>", unsafe_allow_html=True)

    filtered = data[
        data["age_group"].astype(str).isin(selected_ages)
        & data["race"].isin(selected_races)
        & data["6th_stage"].isin(selected_stages)
        & data["status"].isin(selected_statuses)
    ].copy()
    st.markdown("<div class='hero'><div class='eyebrow'>Healthcare data analytics</div><h1>Breast Cancer Patient Dashboard</h1><p>Explore patient demographics, tumor characteristics, and recorded survival patterns.</p></div>", unsafe_allow_html=True)
    if not PLOTLY_AVAILABLE:
        st.info("Plotly is not installed in this environment. The dashboard is using Streamlit chart fallbacks; install Plotly for richer interactive charts: `pip install plotly`.")
    if filtered.empty:
        st.warning("No records match the current filters. Update the sidebar selections to continue.")
        return

    if section == "Overview":
        render_overview(filtered, data)
    elif section == "Patient demographics":
        render_demographics(filtered)
    elif section == "Cancer characteristics":
        render_characteristics(filtered)
    elif section == "Survival analysis":
        render_survival(filtered)
    else:
        render_insights(data, filtered)

    st.markdown("<div class='footnote'>Data source: Breast_cancer.csv · Descriptive analytics for academic exploration.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
