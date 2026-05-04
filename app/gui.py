from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Mini-project 3 ISA RecSys", page_icon="🎯", layout="wide")
st.title("🎯 Mini-project 3: Intelligent Recommender System Application")
st.caption("Production-style GUI for deploying and evaluating a RecSys model from Mini-project 1/2.")


def api_get(path: str) -> Dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

with st.sidebar:
    st.header("Controls")
    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("API and model are running")
    else:
        st.error(health.get("detail", "API error"))
    st.markdown("**Expected CSV schema:** `user_id,item_id,rating` plus optional `title,genres,category,description`.")
    upload = st.file_uploader("Train/deploy another CSV dataset", type=["csv"])
    if upload and st.button("Train and deploy uploaded model"):
        files = {"file": (upload.name, upload.getvalue(), "text/csv")}
        res = requests.post(f"{API_URL}/train", files=files, timeout=120)
        if res.ok:
            st.success("New model trained and deployed.")
            st.json(res.json().get("metrics", {}))
        else:
            st.error(res.text)

metrics = api_get("/metrics")
users = api_get("/users").get("users", [])

rec_tab, eval_tab, risk_tab, docs_tab = st.tabs(["Recommendations", "Quality evaluation", "Risk assessment", "Manual"])

with rec_tab:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.selectbox("Choose user", users, index=0 if users else None) if users else st.text_input("User ID", "U001")
    with col2:
        n = st.slider("Number of recommendations", 1, 20, 10)
    with col3:
        include_seen = st.checkbox("Include already rated/seen items", value=False)

    if st.button("Get recommendations", type="primary"):
        try:
            result = api_post("/recommend", {"user_id": str(user_id), "n": n, "include_seen": include_seen})
            recs = pd.DataFrame(result["recommendations"])
            if recs.empty:
                st.warning("No recommendations found for this user. Try including seen items or retrain with more data.")
            else:
                st.dataframe(recs, use_container_width=True, hide_index=True)
                fig = px.bar(recs, x="item_id", y="predicted_rating", hover_data=[c for c in recs.columns if c not in ["rank"]])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not generate recommendations: {exc}")

    st.subheader("Feedback loop prototype")
    with st.form("feedback"):
        f_col1, f_col2, f_col3 = st.columns(3)
        f_user = f_col1.text_input("User", value=str(user_id) if user_id else "")
        f_item = f_col2.text_input("Item ID")
        f_rating = f_col3.slider("Rating", 1.0, 5.0, 4.0, 0.5)
        comment = st.text_area("Comment", placeholder="Why was this useful or not useful?")
        submitted = st.form_submit_button("Save feedback")
        if submitted:
            if f_user and f_item:
                st.success(api_post("/feedback", {"user_id": f_user, "item_id": f_item, "rating": f_rating, "comment": comment})["message"])
            else:
                st.error("User and Item ID are required.")

with eval_tab:
    st.subheader("Model quality dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", metrics.get("rmse", "n/a"))
    c2.metric("MAE", metrics.get("mae", "n/a"))
    c3.metric("Precision@10", metrics.get("precision_at_k", "n/a"))
    c4.metric("Catalog coverage", metrics.get("catalog_coverage", "n/a"))
    st.write("Dataset/model size")
    st.dataframe(pd.DataFrame([{k: metrics.get(k) for k in ["n_users", "n_items", "n_interactions", "popularity_bias"]}]), hide_index=True, use_container_width=True)
    st.info("Interpretation: lower RMSE/MAE means better rating prediction; higher precision@10 and coverage mean better useful/diverse recommendations.")

with risk_tab:
    st.subheader("Quality evaluation and risk assessment")
    risks = api_get("/risk-assessment")
    st.dataframe(pd.DataFrame(risks["risks"]), hide_index=True, use_container_width=True)
    st.success(risks["improvement_proposal"])

with docs_tab:
    st.markdown("""
    ### Installation manual
    1. Install Docker Desktop / Docker Engine.
    2. Open a terminal in this project folder.
    3. Run: `docker compose up --build`.
    4. Open the GUI at `http://localhost:8501`; API docs are at `http://localhost:8000/docs`.

    ### User manual
    - Use **Recommendations** to choose a user and generate top-N recommendations.
    - Use **Train/deploy another CSV dataset** in the sidebar to replace the sample model with your Mini-project 1 dataset/model pipeline.
    - Use **Quality evaluation** to present RMSE, MAE, Precision@10, coverage, and dataset size.
    - Use **Risk assessment** to discuss cold-start, popularity bias, privacy, drift, and explainability.
    - Use the feedback form as the proposed improvement loop for future retraining.
    """)
