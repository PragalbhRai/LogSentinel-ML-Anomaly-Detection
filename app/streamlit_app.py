"""Run locally with: python -m streamlit run app/streamlit_app.py."""

import os

import streamlit as st

from app.design import apply_design, brand, hero
from app.review_data import DEFAULT_RUN, load_review_data
from app.review_views import anomaly_detection, model_comparison, overview
from app.routes import PAGES
from logsentinel.artifacts import DEFAULT_ARTIFACT_DIR, ArtifactError


def main():
    st.set_page_config(page_title="LogSentinel", page_icon="LS", layout="wide")
    apply_design()
    keys = {page.key for page in PAGES}
    if st.session_state.get("workspace") not in keys:
        requested = st.query_params.get("view", "overview")
        st.session_state.workspace = requested if requested in keys else "overview"
    with st.sidebar:
        brand()
        with st.container(key="ls_navigation"):
            selected = st.radio("Pages", [page.key for page in PAGES],
                                format_func=lambda key: next(p.title for p in PAGES if p.key == key),
                                key="workspace", label_visibility="collapsed")
        st.divider()
        st.caption("HDFS log anomaly detection")
        st.caption("College project · Saved experiment results")
    page = next(p for p in PAGES if p.key == selected)
    st.query_params["view"] = selected
    hero(page.title, page.description)
    try:
        with st.spinner("Checking saved results…"):
            data = load_review_data(os.environ.get("LOGSENTINEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)),
                                    os.environ.get("LOGSENTINEL_EXPERIMENT_DIR", str(DEFAULT_RUN)))
    except (ArtifactError, OSError, ValueError, KeyError) as exc:
        st.error("Saved results could not be verified. Restore the original artifacts and experiment outputs.")
        with st.expander("Error details"):
            st.code(str(exc))
        if st.button("Retry verification"):
            load_review_data.clear()
            st.rerun()
        st.stop()
    {"overview": overview, "model-comparison": model_comparison,
     "anomaly-detection": anomaly_detection}[selected](data)
    st.divider()
    st.caption("LogSentinel · Historical HDFS data · Verified saved results")


if __name__ == "__main__":
    main()
