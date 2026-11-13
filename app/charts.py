"""Plotly views of verified data. No model fitting or scoring lives here."""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

from app.design import COLORS

METHOD_LABELS = {"hmm": "HMM · 3 states", "frequency": "Event frequency", "markov": "First-order Markov"}
METHOD_COLORS = {"hmm": COLORS["cyan"], "frequency": COLORS["teal"], "markov": COLORS["slate"]}
LABEL_COLORS = {"Normal": COLORS["slate"], "Anomaly": COLORS["coral"]}


def theme(figure, height=330):
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, sans-serif", color=COLORS["text"], size=12),
        margin=dict(l=15, r=25, t=20, b=15), height=height,
        legend=dict(orientation="h", y=1.18, x=0), hoverlabel=dict(bgcolor=COLORS["raised"]),
    )
    figure.update_xaxes(gridcolor=COLORS["border"], zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor=COLORS["border"], zeroline=False, automargin=True)
    return figure


def score_distribution(table, threshold, bins=60, density=False):
    edges = np.linspace(float(table.hmm_score.min()), float(table.hmm_score.max()), bins + 1)
    figure = go.Figure()
    for label in ("Normal", "Anomaly"):
        count, _ = np.histogram(table.loc[table.Label == label, "hmm_score"], bins=edges)
        values = count / max(count.sum(), 1) / np.diff(edges) if density else count
        figure.add_trace(go.Bar(x=(edges[:-1]+edges[1:])/2, y=values, width=np.diff(edges),
                               name=f"Actual {label}", marker_color=LABEL_COLORS[label], opacity=.75,
                               hovertemplate="Score bin: %{x:.3f}<br>Value: %{y}<extra>%{fullData.name}</extra>"))
    figure.add_vline(x=threshold, line_color=COLORS["amber"], line_dash="dash",
                     annotation_text="Frozen threshold", annotation_position="top right")
    figure.update_layout(barmode="overlay", xaxis_title="HMM mean negative log-likelihood / event",
                         yaxis_title="Within-label density" if density else "Block count")
    return theme(figure)


def activity_chart(activity):
    figure = go.Figure()
    for name, color, dash in [("Ground-truth anomalous", COLORS["coral"], "dot"),
                               ("Predicted anomalous", COLORS["cyan"], "solid"),
                               ("False alarms", COLORS["amber"], "dash")]:
        figure.add_trace(go.Scatter(x=activity.index, y=activity[name], name=name,
                                   mode="lines+markers", line=dict(color=color, dash=dash, width=2),
                                   hovertemplate="Block-start bin: %{x}<br>Blocks: %{y}<extra>%{fullData.name}</extra>"))
    figure.update_layout(xaxis_title="Observed block start · dataset timezone unspecified", yaxis_title="Block count")
    return theme(figure, 300)


def sequence_chart(table):
    figure = go.Figure(go.Scatter(
        x=table.Position, y=table["Event ID"], mode="lines+markers",
        line=dict(color=COLORS["border"], width=1),
        marker=dict(size=10, color=np.where(table["Unusual transition"], COLORS["coral"],
                                            np.where(table["Repeated event"], COLORS["teal"], COLORS["cyan"])),
                    symbol=np.where(table["Repeated event"], "diamond", "circle")),
        customdata=table[["Template", "Occurrence", "Markov transition P"]],
        hovertemplate="Position %{x} · E%{y}<br>%{customdata[0]}<br>Occurrence %{customdata[1]}<br>Markov P: %{customdata[2]:.6f}<extra></extra>",
    ))
    figure.update_layout(xaxis_title="Event position · observed order", yaxis_title="Categorical event ID (not magnitude)")
    figure.update_xaxes(dtick=1 if len(table) < 25 else None, rangeslider_visible=len(table) > 30)
    return theme(figure, 350)


def heatmap(values, x, y, *, x_title="", y_title="", height=330, annotate=False):
    figure = go.Figure(go.Heatmap(z=values, x=x, y=y, zmin=0, zmax=1,
                                 colorscale=[[0, COLORS["surface"]], [.3, "#22687C"], [1, COLORS["cyan"]]],
                                 texttemplate="%{z:.3f}" if annotate else None,
                                 hovertemplate="%{y} → %{x}<br>Probability: %{z:.6f}<extra></extra>"))
    figure.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return theme(figure, height)


def confusion_chart(metrics, title):
    matrix = [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    figure = go.Figure(go.Heatmap(z=matrix, x=["Predicted normal", "Predicted anomalous"],
                                 y=["Actual normal", "Actual anomalous"], colorscale=[[0, COLORS["surface"]], [1, "#22687C"]],
                                 texttemplate="%{z:,}", showscale=False,
                                 hovertemplate="%{y}<br>%{x}<br>Blocks: %{z:,}<extra></extra>"))
    figure.update_layout(title=dict(text=title, font_size=14))
    return theme(figure, 270)


def precision_recall_chart(table):
    figure = go.Figure()
    y = table.Label.eq("Anomaly").astype(int)
    for method in METHOD_LABELS:
        precision, recall, _ = precision_recall_curve(y, table[method+"_score"])
        figure.add_trace(go.Scatter(x=recall, y=precision, name=METHOD_LABELS[method], mode="lines",
                                   line=dict(color=METHOD_COLORS[method], width=2, shape="hv"),
                                   hovertemplate="Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra>%{fullData.name}</extra>"))
    figure.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    figure.update_xaxes(range=[0, 1]); figure.update_yaxes(range=[0, 1.03])
    return theme(figure, 390)
