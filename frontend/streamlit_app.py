from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Mini-project 3 | Goodbooks Recommender",
    page_icon="📚",
    layout="wide",
)


def api_get(path: str, **params) -> dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def render_items(items: list[dict], strategy: str, mode: str, user_input: str) -> None:
    st.subheader(strategy)
    if not items:
        st.warning(
            "No recommendations found. Try another title/tag, or check that the active books_model.csv is compatible with the cosine matrix."
        )
        return

    df = pd.DataFrame(items)
    visible_df = df.drop(columns=["tags_string"], errors="ignore")
    rename_map = {
        "average_rating": "avg_rating",
        "top_rated_score": "top_rated_score",
        "similarity": "cosine_similarity",
        "hybrid_score": "hybrid_score",
    }
    visible_df = visible_df.rename(columns=rename_map)
    st.dataframe(visible_df, use_container_width=True, hide_index=True)

    for i, row in df.iterrows():
        with st.expander(f"#{i + 1}: {row['title']} — {row.get('authors', 'unknown')}"):
            st.write(f"**Average rating:** {row.get('average_rating', 'N/A')}")
            if "similarity" in row:
                st.write(f"**Cosine similarity:** {row['similarity']}")
            if "hybrid_score" in row:
                st.write(f"**Hybrid reranking score:** {row['hybrid_score']}")
            if "top_rated_score" in row and "similarity" not in row:
                st.write(f"**Top-rated score:** {row['top_rated_score']}")
            if "ratings_count" in row:
                st.write(f"**Ratings count:** {row['ratings_count']}")
            st.write(f"**Tags:** {row.get('tags_string', '')}")
            cols = st.columns([1, 1, 4])
            if cols[0].button("Useful", key=f"u_{i}_{row['title']}"):
                api_post(
                    "/feedback",
                    {
                        "mode": mode,
                        "user_input": user_input,
                        "title": row["title"],
                        "decision": "useful",
                    },
                )
                st.success("Feedback saved.")
            if cols[1].button("Not useful", key=f"n_{i}_{row['title']}"):
                api_post(
                    "/feedback",
                    {
                        "mode": mode,
                        "user_input": user_input,
                        "title": row["title"],
                        "decision": "not_useful",
                    },
                )
                st.success("Feedback saved.")


st.title("📚 Goodbooks Recommender")
st.caption(
    "Active model: cleaned book tag profiles → TF‑IDF max_features=5000 → cosine similarity. "
    "For similarity-based modes, recommendations can be reranked with average_rating."
)

with st.sidebar:
    st.header("Model controls")
    n = st.slider("Number of recommendations", 1, 30, 10)
    hybrid = st.toggle(
        "Use average-rating reranking",
        value=True,
        help="For similarity-based modes, blend cosine similarity with normalized average_rating. Cold-start modes are already top-rated by average_rating.",
    )
    if hybrid:
        sim_w = st.slider(
            "Content similarity weight",
            0.0,
            1.0,
            0.8,
            0.05,
            help="Rating weight is automatically computed as 1 - content weight to avoid invalid weight combinations.",
        )
        rating_w = round(1.0 - sim_w, 4)
        st.caption(f"Effective formula: {sim_w:.2f} × similarity + {rating_w:.2f} × normalized average_rating")
    else:
        sim_w = 1.0
        rating_w = 0.0
        st.caption("Pure cosine similarity for title/profile modes; cold-start modes use average_rating.")

    st.divider()
    st.header("Upload books CSV")
    uploaded = st.file_uploader(
        "books_model.csv",
        type=["csv"],
        help=(
            "Required columns: record_id, goodreads_book_id, title, authors, average_rating, tags_string. "
            "ratings_count is optional. If you use a precomputed cosine matrix, the CSV row order must match it."
        ),
    )
    if uploaded and st.button("Activate uploaded CSV"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        r = requests.post(f"{API_URL}/upload-books-model", files=files, timeout=60)
        if r.ok:
            st.success("Uploaded CSV activated and TF-IDF rebuilt.")
        else:
            st.error(r.text)
    if st.button("Reset configured model"):
        api_post("/reset-sample-model", {})
        st.success("Configured model restored.")

try:
    info = api_get("/model-info")
except Exception as e:
    st.error(
        f"API is not available at {API_URL}. Start the app with docker compose up --build. Error: {e}"
    )
    st.stop()

tabs = st.tabs(["Recommend", "Model & Evaluation", "Risk Assessment", "Manuals"])

with tabs[0]:
    mode_label = st.radio(
        "Recommendation mode",
        [
            "Similar books by title",
            "Cold start: top-rated books",
            "Cold start: top-rated by tag",
            "User profile from favorite books",
            "Smart mode",
        ],
        horizontal=True,
    )
    payload = {
        "n": n,
        "hybrid": hybrid,
        "similarity_weight": sim_w,
        "rating_weight": rating_w,
    }
    user_input = ""

    if mode_label == "Similar books by title":
        q = st.text_input("Search title", value="Harry Potter")
        suggestions = api_get("/titles", q=q, limit=30)["titles"]
        title = st.selectbox("Book title", suggestions or [q])
        payload.update({"mode": "title", "title": title})
        user_input = title
    elif mode_label == "Cold start: top-rated books":
        st.info("No user history is needed. The app ranks books by average_rating because the current export has no real ratings_count column.")
        payload.update({"mode": "global"})
        user_input = "top-rated"
    elif mode_label == "Cold start: top-rated by tag":
        genre = st.text_input("Tag/genre", value="fantasy")
        payload.update({"mode": "genre", "genre": genre})
        user_input = genre
    elif mode_label == "User profile from favorite books":
        q = st.text_input("Filter favorite-book list", value="Harry Potter")
        suggestions = api_get("/titles", q=q, limit=30)["titles"]
        favs = st.multiselect(
            "Favorite books",
            suggestions,
            default=suggestions[:3] if len(suggestions) >= 3 else suggestions,
        )
        payload.update({"mode": "profile", "favorite_titles": favs})
        user_input = "; ".join(favs)
    else:
        auto_text = st.text_input(
            "Input: empty = top-rated, short text = tag, exact title = similar books, or use profile below",
            value="The Hunger Games (The Hunger Games, #1)",
        )
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
    st.header("Active deployed model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", info["books"])
    c2.metric("TF‑IDF rows", info["tfidf_shape"][0])
    c3.metric("TF‑IDF features", info["tfidf_shape"][1])
    if info.get("cosine_sim_loaded_from_file"):
        st.success("Precomputed cosine_sim_best.npy is loaded from models/.")
    else:
        st.warning("cosine_sim_best.npy was not loaded; the API computed cosine similarity at startup.")
    if info.get("has_ratings_count"):
        st.info("The active CSV contains ratings_count. The UI displays it, but current cold-start ranking still uses average_rating for clarity.")
    else:
        st.info("The active CSV has no ratings_count, so cold-start ranking is top-rated by average_rating, not popularity-based.")
    st.json({k: v for k, v in info.items() if k not in {"best_experiment"}})

    st.subheader("Benchmark from Mini-project 1 notebook")
    experiments = pd.DataFrame(api_get("/experiments")["experiments"])
    if not experiments.empty:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
        fig = px.bar(
            experiments,
            x="Experiment",
            y=["Precision@10", "Recall@10", "F1@10"],
            barmode="group",
            title="Experiment comparison",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(
            "Selected deployment model: exp5_maxfeat — TF‑IDF on tags_string with max_features=5000. "
            "It is used together with optional average-rating reranking."
        )
    st.markdown(
        """
        **Quality evaluation approach:** leave-one-out on sampled users. For each user,
        ratings ≥ 4 are relevant books; one is used as query and the rest are ground truth.
        Metrics: Precision@K, Recall@K, F1@K.
        """
    )

with tabs[2]:
    st.header("Quality risks and deployment risks")
    st.markdown(
        """
        | Risk | Impact | Mitigation in this app |
        |---|---|---|
        | Cold-start user has no history | No personalized vector | Top-rated fallback and tag-based top-rated fallback |
        | The current full export has no `ratings_count` | We cannot measure true popularity/confidence | UI labels the fallback as **top-rated**, not popularity-based |
        | Rating-only cold start can favor niche books with high average rating | Some results may be less broadly trusted | Future improvement: add `ratings_count` from the original Goodbooks `books.csv` |
        | Noisy Goodreads tags like `to-read`, `favorites` | Bad content representation | MP1 preprocessing removes filler tags and uses cleaned tag profiles |
        | Hybrid score can be misinterpreted | Users may set unclear weights | UI uses one content-weight slider; backend normalizes weights |
        | Sparse user histories in evaluation | Low absolute Precision@10 | Metrics are interpreted against the 10k-book catalog and qualitative tests are included |
        | Duplicate or confusing IDs | Wrong joins or wrong book ordering | `record_id` is kept stable and `cosine_sim_best.npy` must match the CSV row order |
        | Memory use of full cosine matrix | 10k × 10k matrix is large | API loads it once via `mmap_mode`; for larger datasets use top-N export or FAISS |
        | Feedback not used for retraining yet | No online learning from users | Feedback buttons save data for future supervised reranking |
        """
    )
    st.info(
        "Improvement proposal for 3.1.C: add ratings_count to the export, use saved useful/not-useful feedback to learn a reranker, personalize tag weights, and monitor recommendation drift."
    )

with tabs[3]:
    st.header("Installation manual")
    st.code("docker compose up --build", language="bash")
    st.write("Open GUI at http://localhost:8501 and API docs at http://localhost:8000/docs.")
    st.header("User manual")
    st.markdown(
        """
        1. Choose a recommendation mode.
        2. Select a book title, tag/genre, or several favorite books.
        3. Adjust the number of recommendations.
        4. Optionally enable average-rating reranking and choose the content-similarity weight.
        5. Click **Generate recommendations**.
        6. Mark recommendations as useful/not useful to collect feedback.
        7. Use `/model-info` or the Model tab to verify that `cosine_sim_best.npy` is loaded.
        """
    )
