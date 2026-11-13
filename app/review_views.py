"""Simple presentation of real saved results, without training or tuning."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.design import COLORS, metric_card
from app.review_data import MODEL_NAMES


def bar_chart(labels, values, colors, *, score=False):
    figure = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.4f}" if score else f"{v:,}" for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    figure.update_layout(
        height=285, margin=dict(l=0, r=65, t=10, b=15), bargap=0.5,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=14, color=COLORS["text"]),
        xaxis=dict(title="F1-score" if score else "Number of blocks", gridcolor=COLORS["border"],
                   zeroline=False, range=[0, 1.08] if score else None),
        yaxis=dict(autorange="reversed", automargin=True), showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def overview(data):
    st.markdown("LogSentinel uses Machine Learning to identify unusual activity in **HDFS**, "
                "a system that stores files across many computers. It groups log messages by "
                "data block and classifies each block as **Normal** or **Anomaly**.")
    st.markdown("This dashboard explains the dataset, compares three trained models, "
                "and lets you check their saved predictions.")
    for column, (label, key, detail, accent) in zip(st.columns(3), (
            ("Total dataset blocks", "total", "All labeled HDFS blocks", "cyan"),
            ("Normal blocks", "normal", "Expected activity", "teal"),
            ("Anomalous blocks", "anomaly", "Unusual activity", "coral"))):
        with column:
            metric_card(label, f"{data.counts[key]:,}", detail, accent)
    st.subheader("Most blocks are normal")
    bar_chart(["Normal", "Anomaly"], [data.counts["normal"], data.counts["anomaly"]],
              [COLORS["teal"], COLORS["coral"]])
    st.caption("This class imbalance makes finding the smaller anomaly group more challenging.")
    st.caption(f"Dataset totals include all labeled blocks. The experiments use {data.retained_blocks:,} "
               "blocks after excluding blocks that cross the training, validation or test time boundaries.")


def model_comparison(data):
    best = data.comparison.loc[data.comparison.f1.idxmax()]
    st.success(f"{MODEL_NAMES[best.model]} has the highest test F1-score: {best.f1:.6f}.")
    st.caption(f"All three models were evaluated on the same {len(data.test):,} saved test blocks.")
    table = pd.DataFrame({
        "Model": data.comparison.model.map(MODEL_NAMES),
        "Precision": data.comparison.precision.map(lambda v: f"{v:.4f}"),
        "Recall": data.comparison.recall.map(lambda v: f"{v:.4f}"),
        "F1-score": data.comparison.f1.map(lambda v: f"{v:.4f}"),
        "False-alarm rate": data.comparison.false_alarm_rate.map(lambda v: f"{v:.4%}"),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Precision: how often an alert is correct. Recall: how many anomalies are found. "
               "F1 balances both. A lower false-alarm rate is better.")
    st.subheader("F1-score comparison")
    bar_chart(["HMM", "Class-weighted RF", "SMOTE RF"], data.comparison.f1.tolist(),
              [COLORS["slate"], COLORS["cyan"], COLORS["teal"]], score=True)
    st.markdown("**Why SMOTE?** It creates extra examples of the smaller anomaly class "
                "**only in the training data**. Validation and test data were left unchanged.")
    st.caption("The Random Forest models learn from labeled normal and anomalous training blocks; "
               "the HMM was trained on normal blocks.")
    st.info("Limitation: 99.90% of test blocks share event-count patterns with training, "
            "so results on completely new patterns may be lower.")


def anomaly_detection(data):
    st.markdown("Choose a test block and select **Predict** to see the result from each model.")
    st.caption("This replays verified saved predictions. It does not train models or simulate new results.")
    search = st.text_input("Find a block", placeholder="Type all or part of a block ID", key="block_search")
    matches = data.test[data.test.BlockId.str.contains(search.strip(), case=False, regex=False)]
    if matches.empty:
        st.info("No saved test blocks match. Try another block ID.")
        return
    options = matches.BlockId.head(200).tolist()
    if st.session_state.get("test_block") not in options:
        st.session_state.test_block = options[0]
    block = st.selectbox("Saved test block", options, key="test_block")
    if len(matches) > len(options):
        st.caption(f"Showing the first {len(options)} of {len(matches):,} matching blocks. Search by ID to find any test block.")
    if st.button("Predict", type="primary", key="predict"):
        st.session_state.review_prediction = (block, data.revision)
    if st.session_state.get("review_prediction") != (block, data.revision):
        return
    row = data.test.loc[block]
    st.subheader("Prediction results")
    st.markdown(f"**Actual label: {row.Label}**")
    results = pd.DataFrame([
        {"Model": label, "Prediction": "Anomaly" if row[method + "_prediction"] else "Normal",
         "Actual label": row.Label} for method, label in MODEL_NAMES.items()
    ])
    st.dataframe(results, hide_index=True, width="stretch")
    st.caption("Actual labels are the dataset's ground truth. Predictions are the models' saved decisions.")
