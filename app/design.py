"""Shared visual tokens, escaped components and Plotly presentation defaults."""

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "canvas": "#080F1C", "surface": "#101C2D", "raised": "#17263A",
    "text": "#E8F0FA", "muted": "#A9B9CE", "cyan": "#50D9F5",
    "teal": "#7CD5BC", "amber": "#F0C078", "coral": "#FF938C",
    "border": "#2A3C53", "slate": "#A8BBD5",
}
MODEL_COLORS = {"frequency": COLORS["teal"], "markov": COLORS["slate"],
                "HMM_3state": COLORS["cyan"]}
MODEL_NAMES = {"frequency": "Event frequency", "markov": "First-order Markov", "HMM_3state": "Categorical HMM · 3 states"}


def apply_design():
    variables = ";".join(f"--ls-{key}:{value}" for key, value in COLORS.items())
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>:root{{{variables}}}{css}</style>", unsafe_allow_html=True)


def brand():
    st.markdown('<div class="ls-brand"><span class="ls-monogram" aria-hidden="true">LS</span>'
                '<div>LogSentinel<small>LOG ANOMALY DETECTION</small></div></div>', unsafe_allow_html=True)


def hero(title, description):
    st.markdown('<div class="ls-hero"><div class="ls-eyebrow">HDFS · MACHINE LEARNING PROJECT</div>'
                f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def metric_card(label, value, detail, accent="cyan"):
    if accent not in COLORS:
        raise ValueError("Unknown visual accent.")
    st.markdown(f'<article class="ls-metric" style="--ls-accent:var(--ls-{accent})">'
                f'<div class="ls-metric-label">{escape(label)}</div><div class="ls-metric-value">{escape(value)}</div>'
                f'<div class="ls-metric-detail">{escape(detail)}</div></article>', unsafe_allow_html=True)


def empty_state(title, description):
    st.markdown(f'<div class="ls-empty" role="status"><h3>{escape(title)}</h3>'
                f'<p>{escape(description)}</p></div>', unsafe_allow_html=True)


def error_state():
    st.error("The experiment bundle could not be verified. Restore the original exports and retry.", icon="⚠️")
    st.caption("No experimental data is displayed until bundle verification succeeds.")


def model_comparison_chart(comparison):
    ordered = comparison.sort_values("f1", ascending=True)
    figure = go.Figure(go.Bar(
        x=ordered.f1, y=[MODEL_NAMES[m] for m in ordered.model], orientation="h",
        marker_color=[MODEL_COLORS[m] for m in ordered.model],
        text=[f"{v:.4f}" for v in ordered.f1], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Test F1: %{x:.6f}<extra></extra>",
    ))
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, sans-serif", color=COLORS["text"], size=13),
        margin=dict(l=0, r=50, t=10, b=0), height=255, bargap=0.48,
        xaxis=dict(range=[0, 1], title="Test F1 · fixed validation thresholds", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(title=None, automargin=True), showlegend=False,
    )
    return figure
