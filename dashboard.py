import streamlit as st
import pandas as pd
import plotly.express as px


CUSTOM_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="white", size=14),
    title_font=dict(size=22),
    margin=dict(l=20, r=20, t=50, b=20),
)
EDA_LAYOUT = {
    **CUSTOM_LAYOUT,
    "autosize": True,
    "margin": dict(l=65, r=30, t=75, b=65),
}
HISTORY_LAYOUT = {
    **CUSTOM_LAYOUT,
    "autosize": True,
    "font": dict(color="white", size=13),
    "title_font": dict(size=18),
    "margin": dict(l=70, r=30, t=70, b=105),
}

PCOS_LABELS = ["No PCOS", "PCOS"]
PCOS_COLORS = {"No PCOS": "#00CC96", "PCOS": "#EF553B"}
SCORE_BAND_COLORS = {
    "Low": "#06B6D4",
    "Moderate": "#3B82F6",
    "High": "#8B5CF6",
}
PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}
SYMPTOM_FEATURES = {
    "irregular_periods": "Irregular periods",
    "acne": "Acne",
    "hair_growth": "Excess hair growth",
    "weight_gain": "Recent weight gain",
    "skin_darkening": "Skin darkening",
    "fast_food": "Frequent fast food",
    "exercise": "Regular exercise",
}


def render_history(history):
    st.markdown("### Session summary")

    if not history:
        st.info("No session predictions yet.")
        return

    df = pd.DataFrame(history)
    df["session_prediction"] = range(1, len(df) + 1)

    st.caption(
        "Each point is an individual experimental symptom-model output, not a "
        "longitudinal medical trend. Scores are not clinical PCOS probabilities, "
        "and score bands are not medically validated."
    )

    latest_score = df["risk_pct"].iloc[-1]
    latest_bmi = df["bmi"].iloc[-1]
    latest_level = df["level"].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest experimental score", f"{latest_score}%")
    c2.metric("Latest input BMI", f"{latest_bmi:.2f} kg/m²")
    c3.metric("Experimental score band", latest_level)

    st.markdown("---")
    score_fig = px.scatter(
        df,
        x="session_prediction",
        y="risk_pct",
        color="level",
        category_orders={"level": list(SCORE_BAND_COLORS)},
        color_discrete_map=SCORE_BAND_COLORS,
        labels={
            "session_prediction": "Prediction number",
            "risk_pct": "Experimental symptom-model score (%)",
            "level": "Experimental score band",
        },
        title="Individual model outputs",
    )
    score_fig.update_traces(
        marker=dict(size=11, line=dict(width=1, color="#0E1117")),
        hovertemplate=(
            "Prediction %{x:.0f}<br>"
            "Experimental score %{y:.1f}%<br>"
            "Band %{fullData.name}<extra></extra>"
        ),
    )
    score_fig.update_layout(
        **HISTORY_LAYOUT,
        height=390,
        xaxis=dict(
            title="Prediction number",
            type="linear",
            tickmode="linear",
            dtick=1,
            tickformat="d",
            range=[0.5, len(df) + 0.5],
            automargin=True,
        ),
        yaxis=dict(
            title="Experimental symptom-model score (%)",
            range=[0, 100],
            dtick=20,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            title_text="Experimental score band",
        ),
    )
    st.plotly_chart(score_fig, width="stretch", config=PLOTLY_CONFIG)

    left_col, right_col = st.columns(2)

    with left_col:
        level_counts = (
            df["level"]
            .value_counts()
            .reindex(list(SCORE_BAND_COLORS), fill_value=0)
        )
        level_counts = (
            level_counts[level_counts > 0]
            .rename_axis("Experimental score band")
            .reset_index(name="Prediction count")
        )
        level_fig = px.bar(
            level_counts,
            x="Experimental score band",
            y="Prediction count",
            color="Experimental score band",
            text="Prediction count",
            category_orders={
                "Experimental score band": list(SCORE_BAND_COLORS)
            },
            color_discrete_map=SCORE_BAND_COLORS,
            labels={
                "Experimental score band": "Experimental band",
                "Prediction count": "Number of predictions",
            },
            title="Counts by experimental score band",
        )
        level_fig.update_layout(
            **HISTORY_LAYOUT,
            height=330,
            showlegend=False,
            xaxis=dict(automargin=True),
            yaxis=dict(rangemode="tozero", automargin=True),
        )
        level_fig.update_layout(margin=dict(l=65, r=25, t=65, b=70))
        level_fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(level_fig, width="stretch", config=PLOTLY_CONFIG)

    with right_col:
        bmi_fig = px.bar(
            df,
            x="session_prediction",
            y="bmi",
            labels={
                "session_prediction": "Prediction number",
                "bmi": "Input BMI (kg/m²)",
            },
            title="BMI values entered for each prediction",
        )
        bmi_fig.update_layout(
            **HISTORY_LAYOUT,
            height=330,
            xaxis=dict(
                title="Prediction number",
                type="linear",
                tickmode="linear",
                dtick=1,
                tickformat="d",
                range=[0.5, len(df) + 0.5],
                automargin=True,
                tickangle=-35,
            ),
            yaxis=dict(title="Input BMI (kg/m²)", rangemode="tozero", automargin=True),
        )
        bmi_fig.update_layout(margin=dict(l=65, r=25, t=65, b=100))
        st.plotly_chart(bmi_fig, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("---")
    scatter_fig = px.scatter(
        df,
        x="bmi",
        y="risk_pct",
        color="level",
        category_orders={"level": list(SCORE_BAND_COLORS)},
        color_discrete_map=SCORE_BAND_COLORS,
        hover_data={"session_prediction": True, "level": True},
        labels={
            "bmi": "Input BMI (kg/m²)",
            "risk_pct": "Experimental symptom-model score (%)",
            "level": "Experimental score band",
            "session_prediction": "Prediction number",
        },
        title="Input BMI versus individual model output",
    )
    scatter_fig.update_traces(
        marker=dict(size=11, line=dict(width=1, color="#0E1117")),
        hovertemplate=(
            "Input BMI %{x:.2f} kg/m²<br>"
            "Experimental score %{y:.1f}%<br>"
            "Prediction %{customdata[0]:.0f}<br>"
            "Band %{customdata[1]}<extra></extra>"
        ),
    )
    scatter_fig.update_layout(
        **HISTORY_LAYOUT,
        height=400,
        xaxis=dict(title="Input BMI (kg/m²)", automargin=True),
        yaxis=dict(
            title="Experimental symptom-model score (%)",
            range=[0, 100],
            dtick=20,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            title_text="Experimental score band",
        ),
    )
    st.plotly_chart(scatter_fig, width="stretch", config=PLOTLY_CONFIG)


def render_eda(df):
    st.markdown("---")
    st.markdown("## Clinical Dataset Exploratory Analysis")

    required_columns = {
        "pcos",
        "age",
        "weight",
        "height",
        "bmi",
        "irregular_periods",
        *SYMPTOM_FEATURES.keys(),
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        st.error(f"Clinical dataset is missing required columns: {missing_columns}")
        return

    eda = df.copy()
    eda["pcos_label"] = pd.to_numeric(
        eda["pcos"], errors="coerce"
    ).map({0: "No PCOS", 1: "PCOS"})
    if eda["pcos_label"].isna().any():
        st.error("PCOS target values must be encoded as 0 or 1.")
        return

    with st.expander("View clinical dataset"):
        st.caption(
            f"{len(eda):,} records | "
            f"{len(SYMPTOM_FEATURES) + 4} clinical features and target"
        )
        st.dataframe(df.head(10), width="stretch")

    class_counts = (
        eda["pcos_label"]
        .value_counts()
        .reindex(PCOS_LABELS, fill_value=0)
        .rename_axis("PCOS label")
        .reset_index(name="Records")
    )

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("### PCOS class distribution")
        class_fig = px.bar(
            class_counts,
            x="PCOS label",
            y="Records",
            color="PCOS label",
            text="Records",
            category_orders={"PCOS label": PCOS_LABELS},
            color_discrete_map=PCOS_COLORS,
            labels={"PCOS label": "Clinical label", "Records": "Record count"},
            title="Clinical records by PCOS label",
        )
        class_fig.update_layout(**EDA_LAYOUT, height=410, showlegend=False)
        st.plotly_chart(class_fig, width="stretch")

    with top_right:
        st.markdown("### Age distribution")
        age_fig = px.histogram(
            eda,
            x="age",
            color="pcos_label",
            barmode="overlay",
            opacity=0.75,
            nbins=20,
            category_orders={"pcos_label": PCOS_LABELS},
            color_discrete_map=PCOS_COLORS,
            labels={"age": "Age (years)", "count": "Record count", "pcos_label": "PCOS label"},
            title="Age distribution by clinical label",
        )
        age_fig.update_layout(
            **EDA_LAYOUT,
            height=440,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.27,
                xanchor="center",
                x=0.5,
                title_text="PCOS label",
            ),
            xaxis_automargin=True,
            yaxis_automargin=True,
        )
        age_fig.update_layout(margin=dict(l=65, r=25, t=75, b=115))
        st.plotly_chart(age_fig, width="stretch")

    middle_left, middle_right = st.columns(2)
    with middle_left:
        st.markdown("### BMI analysis")
        bmi_fig = px.box(
            eda,
            x="pcos_label",
            y="bmi",
            color="pcos_label",
            points="outliers",
            category_orders={"pcos_label": PCOS_LABELS},
            color_discrete_map=PCOS_COLORS,
            labels={
                "pcos_label": "PCOS label",
                "bmi": "BMI (kg/m²)",
            },
            title="BMI distribution by clinical label",
        )
        bmi_fig.update_layout(**EDA_LAYOUT, height=410, showlegend=False)
        st.plotly_chart(bmi_fig, width="stretch")

    with middle_right:
        st.markdown("### Irregular-period distribution")
        period_pattern = pd.to_numeric(
            eda["irregular_periods"], errors="coerce"
        ).map({0: "Regular", 1: "Irregular"})
        period_rates = (
            pd.crosstab(eda["pcos_label"], period_pattern, normalize="index")
            .reindex(index=PCOS_LABELS, columns=["Regular", "Irregular"], fill_value=0)
            .rename_axis("PCOS label")
            .reset_index()
            .melt(
                id_vars="PCOS label",
                var_name="Period pattern",
                value_name="Percentage",
            )
        )
        period_rates["Percentage"] *= 100
        period_fig = px.bar(
            period_rates,
            x="PCOS label",
            y="Percentage",
            color="Period pattern",
            barmode="group",
            category_orders={"PCOS label": PCOS_LABELS},
            color_discrete_map={"Regular": "#636EFA", "Irregular": "#FECB52"},
            labels={
                "PCOS label": "Clinical label",
                "Percentage": "Records within label (%)",
            },
            title="Cycle pattern within each PCOS label",
        )
        period_fig.update_layout(
            **EDA_LAYOUT,
            height=440,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                title_text="Cycle pattern",
            ),
            xaxis_automargin=True,
            yaxis_automargin=True,
        )
        period_fig.update_layout(margin=dict(l=65, r=25, t=75, b=110))
        st.plotly_chart(period_fig, width="stretch")

    st.markdown("---")
    st.markdown("### Symptom frequency by PCOS label")
    numeric_symptoms = eda[list(SYMPTOM_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    symptom_rates = numeric_symptoms.assign(
        pcos_label=eda["pcos_label"]
    ).groupby("pcos_label")[list(SYMPTOM_FEATURES)].mean().mul(100)
    symptom_rates = symptom_rates.reindex(PCOS_LABELS).reset_index().melt(
        id_vars="pcos_label",
        var_name="feature",
        value_name="Frequency (%)",
    )
    symptom_rates["feature"] = symptom_rates["feature"].map(SYMPTOM_FEATURES)
    symptom_fig = px.bar(
        symptom_rates,
        x="feature",
        y="Frequency (%)",
        color="pcos_label",
        barmode="group",
        category_orders={"pcos_label": PCOS_LABELS},
        color_discrete_map=PCOS_COLORS,
        labels={"feature": "Clinical symptom / behavior", "pcos_label": "PCOS label"},
        title="Percentage of records reporting each feature",
    )
    symptom_fig.update_layout(
        **EDA_LAYOUT,
        height=560,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            xanchor="center",
            x=0.5,
            title_text="PCOS label",
        ),
        xaxis_tickangle=-25,
        xaxis_automargin=True,
        yaxis_automargin=True,
    )
    symptom_fig.update_layout(margin=dict(l=75, r=30, t=80, b=145))
    st.plotly_chart(symptom_fig, width="stretch")

    st.markdown("---")
    st.markdown("### Feature correlation heatmap")
    numeric_data = eda.drop(columns=["pcos_label"]).select_dtypes(
        include="number"
    )
    feature_labels = {
        "pcos": "PCOS label (0/1)",
        "age": "Age (years)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "bmi": "BMI",
        "irregular_periods": "Irregular periods (0/1)",
        "acne": "Acne (0/1)",
        "hair_growth": "Excess hair growth (0/1)",
        "weight_gain": "Recent weight gain (0/1)",
        "skin_darkening": "Skin darkening (0/1)",
        "fast_food": "Frequent fast food (0/1)",
        "exercise": "Regular exercise (0/1)",
    }
    correlation = numeric_data.corr().rename(
        index=feature_labels, columns=feature_labels
    ).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if correlation.empty:
        st.info("No numeric clinical features are available for correlation.")
        return

    correlation_fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"x": "Feature", "y": "Feature", "color": "Correlation"},
        title="Pearson correlation across numeric clinical features",
    )
    correlation_fig.update_layout(
        **EDA_LAYOUT,
        height=max(520, len(correlation.columns) * 42),
    )
    st.plotly_chart(correlation_fig, width="stretch")
