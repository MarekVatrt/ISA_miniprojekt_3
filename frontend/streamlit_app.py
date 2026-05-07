from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Mini-project 3 | Goodbooks Recommender", page_icon="📚", layout="wide")


def api_get(path: str, **params) -> dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def render_items(items: list[dict], strategy: str, mode: str, user_input: str):
    st.subheader(strategy)
    if not items:
        st.warning("No recommendations found. Try another title, genre, or upload the full books_model.csv from Mini-project 1.")
        return
    df = pd.DataFrame(items)
    st.dataframe(df.drop(columns=["tags_string"], errors="ignore"), use_container_width=True, hide_index=True)
    for i, row in df.iterrows():
        with st.expander(f"#{i+1}: {row['title']} — {row.get('authors', 'unknown')}"):
            st.write(f"**Average rating:** {row.get('average_rating', 'N/A')}")
            if "similarity" in row:
                st.write(f"**Cosine similarity:** {row['similarity']}")
            if "hybrid_score" in row:
                st.write(f"**Hybrid score:** {row['hybrid_score']}")
            st.write(f"**Tags:** {row.get('tags_string', '')}")
            cols = st.columns([1, 1, 4])
            if cols[0].button("Useful", key=f"u_{i}_{row['title']}"):
                api_post("/feedback", {"mode": mode, "user_input": user_input, "title": row["title"], "decision": "useful"})
                st.success("Feedback saved.")
            if cols[1].button("Not useful", key=f"n_{i}_{row['title']}"):
                api_post("/feedback", {"mode": mode, "user_input": user_input, "title": row["title"], "decision": "not_useful"})
                st.success("Feedback saved.")


st.title("📚 Mini-project 3: Goodbooks Content-Based Recommender")
st.caption("Dockerized deployment of Mini-project 1: cleaned tag profiles → TF‑IDF max_features=5000 → cosine similarity → recommendations.")

with st.sidebar:
    st.header("Model controls")
    n = st.slider("Number of recommendations", 1, 30, 10)
    hybrid = st.toggle("Use hybrid score", value=True, help="80% content similarity + 20% normalized rating by default.")
    sim_w = st.slider("Similarity weight", 0.0, 1.0, 0.8, 0.05)
    rating_w = st.slider("Rating weight", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    st.header("Upload MP1 model")
    uploaded = st.file_uploader("books_model.csv", type=["csv"], help="Columns: record_id, goodreads_book_id, title, authors, average_rating, tags_string; ratings_count optional.")
    if uploaded and st.button("Activate uploaded model"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        r = requests.post(f"{API_URL}/upload-books-model", files=files, timeout=60)
        if r.ok:
            st.success("Uploaded model activated and TF-IDF rebuilt.")
        else:
            st.error(r.text)
    if st.button("Reset sample model"):
        api_post("/reset-sample-model", {})
        st.success("Sample model restored.")

try:
    info = api_get("/model-info")
except Exception as e:
    st.error(f"API is not available at {API_URL}. Start the app with docker compose up --build. Error: {e}")
    st.stop()

tabs = st.tabs(["Recommend", "Model & Evaluation", "Risk Assessment", "Manuals"])

with tabs[0]:
    mode_label = st.radio(
        "Recommendation mode",
        ["Similar books by title", "Cold start: global popularity", "Cold start: genre", "User profile from favorite books", "Auto wrapper"],
        horizontal=True,
    )
    payload = {"n": n, "hybrid": hybrid, "similarity_weight": sim_w, "rating_weight": rating_w}
    user_input = ""

    if mode_label == "Similar books by title":
        q = st.text_input("Search title", value="Harry Potter")
        suggestions = api_get("/titles", q=q, limit=30)["titles"]
        title = st.selectbox("Book title", suggestions or [q])
        payload.update({"mode": "title", "title": title})
        user_input = title
    elif mode_label == "Cold start: global popularity":
        payload.update({"mode": "global"})
        user_input = "global"
    elif mode_label == "Cold start: genre":
        genre = st.text_input("Genre/tag", value="fantasy")
        payload.update({"mode": "genre", "genre": genre})
        user_input = genre
    elif mode_label == "User profile from favorite books":
        q = st.text_input("Filter favorite-book list", value="Harry Potter")
        suggestions = api_get("/titles", q=q, limit=30)["titles"]
        favs = st.multiselect("Favorite books", suggestions, default=suggestions[:3] if len(suggestions) >= 3 else suggestions)
        payload.update({"mode": "profile", "favorite_titles": favs})
        user_input = "; ".join(favs)
    else:
        auto_text = st.text_input("Input: empty/global, short genre, exact book title, or use profile below", value="The Hunger Games (The Hunger Games, #1)")
        use_profile = st.toggle("Use selected favorites as list input")
        if use_profile:
            suggestions = api_get("/titles", q="Harry Potter", limit=10)["titles"]
            favs = st.multiselect("Favorite books", suggestions, default=suggestions[:3])
            payload.update({"mode": "auto", "favorite_titles": favs})
            user_input = "; ".join(favs)
        else:
            payload.update({"mode": "auto", "title": auto_text})
            user_input = auto_text

    if st.button("Generate recommendations", type="primary"):
        try:
            result = api_post("/recommend", payload)
            render_items(result["items"], result["strategy"], payload["mode"], user_input)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

with tabs[1]:
    st.header("Deployed Mini-project 1 model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", info["books"])
    c2.metric("TF‑IDF rows", info["tfidf_shape"][0])
    c3.metric("TF‑IDF features", info["tfidf_shape"][1])
    st.json({k: v for k, v in info.items() if k not in {"best_experiment"}})
    st.subheader("Benchmark from Mini-project 1 notebook")
    experiments = pd.DataFrame(api_get("/experiments")["experiments"])
    if not experiments.empty:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
        fig = px.bar(experiments, x="Experiment", y=["Precision@10", "Recall@10", "F1@10"], barmode="group", title="Experiment comparison")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Selected deployment model: exp5_maxfeat — TF‑IDF on tags_string with max_features=5000. It had the highest Precision@10 and F1@10 in MP1.")
    st.markdown("""
    **Quality evaluation approach:** leave-one-out on sampled users. For each user, ratings ≥ 4 are relevant books; one is used as query and the rest are ground truth. Metrics: Precision@K, Recall@K, F1@K.
    """)

with tabs[3]:
    st.header("Installation manual")
    st.code("docker compose up --build", language="bash")
    st.write("Open GUI at http://localhost:8501 and API docs at http://localhost:8000/docs.")
    st.header("User manual")
    st.markdown("""
    1. Choose a recommendation mode.
    2. Select a book title, genre, or several favorite books.
    3. Adjust number of recommendations and hybrid score weights.
    4. Click **Generate recommendations**.
    5. Mark recommendations as useful/not useful to collect feedback.
    6. Upload your full `models/books_model.csv` from Mini-project 1 to replace the bundled sample data.
    """)
