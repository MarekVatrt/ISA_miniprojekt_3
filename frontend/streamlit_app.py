from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from datetime import datetime
import uuid
from components.visualizations import (
    plot_similarity_distribution,
    plot_rating_vs_similarity,
)

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


def render_suggestion_buttons(suggestions, key_prefix, n_cols=5):
    """Render suggestions as clickable buttons laid out in n_cols columns.
    Returns the clicked suggestion (string) or None.
    """
    selected = None
    if suggestions:
        cols = st.columns(n_cols)
        for i, s in enumerate(suggestions):
            col = cols[i % n_cols]
            # use a stable unique key per suggestion button
            if col.button(s, key=f"{key_prefix}_sugg_{i}"):
                selected = s
    return selected


def favorites_manager(session_key="favorites_list", input_key="fav_input", suggestion_prefix="fav"):
    """Small UI for building a list of favorite titles.
    - session_key: session_state key where the favorites list is stored
    - input_key: session_state key for the free-text input used to search/add
    Returns the current favorites list.
    """
    # ensure state keys exist
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    changed_flag = f"{session_key}_changed"
    if changed_flag not in st.session_state:
        st.session_state[changed_flag] = False

    clear_flag = f"{input_key}_clear"
    if clear_flag not in st.session_state:
        st.session_state[clear_flag] = False

    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    # if a previous action requested clearing the input, do it before creating widgets
    if st.session_state.get(clear_flag):
        st.session_state[input_key] = ""
        st.session_state[clear_flag] = False

    # Use a single selectbox backed by the full title list (client-side search-as-you-type)
    all_titles = st.session_state.get("all_titles", [])
    options = [""] + all_titles if all_titles else [""]
    sel = st.selectbox("Add favorite (type to filter and pick)", options=options, index=0, key=f"{suggestion_prefix}_global_select")
    if sel:
        if sel not in st.session_state[session_key]:
            st.session_state[session_key].append(sel)
            st.session_state[changed_flag] = True

    # allow adding the typed value explicitly (schedule clear for next run)
    if st.button("Add typed", key=f"{input_key}_add_typed"):
        typed = st.session_state.get(input_key, "").strip()
        if typed and typed not in st.session_state[session_key]:
            st.session_state[session_key].append(typed)
            st.session_state[changed_flag] = True

    # display current favorites with remove buttons
    if st.session_state[session_key]:
        st.write("Current favorites:")
        for i, s in enumerate(list(st.session_state[session_key])):
            c1, c2 = st.columns([8, 1])
            c1.write(s)
            if c2.button("Remove", key=f"{session_key}_remove_{i}"):
                st.session_state[session_key].pop(i)
                st.session_state[changed_flag] = True

    return st.session_state[session_key]


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

    # Visualizations for the recommendation set - show side-by-side
    c1, c2 = st.columns([1, 1])
    with c1:
        try:
            fig1 = plot_similarity_distribution(df)
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.info(f"Similarity histogram not available: {e}")
    with c2:
        try:
            fig2 = plot_rating_vs_similarity(df)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.info(f"Rating vs similarity plot not available: {e}")

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


def run_recommendation(payload: dict, user_input: str) -> None:
    """Call the recommendation API and store the result in session_state.
    This centralizes the behaviour used by both the Generate button and automatic triggers.
    """
    # Basic client-side validation to avoid sending bad requests to the API
    mode = payload.get("mode")
    if mode == "title" and not payload.get("title"):
        st.warning("Please pick a title before requesting recommendations.")
        return
    if mode == "genre" and not payload.get("genre"):
        st.warning("Please enter a genre/tag before requesting recommendations.")
        return
    if mode == "profile" and not payload.get("favorite_titles"):
        st.warning("Please add favorite books to build a profile before requesting recommendations.")
        return

    # Prepare metadata for correlation (timestamp + request id). Do not print immediately
    # — we store these in session_state so the UI can render them below the inputs/results.
    ts = datetime.utcnow().isoformat() + "Z"
    req_id = uuid.uuid4().hex

    # Record the attempted payload so we can inspect what was sent even if the request fails.
    st.session_state["last_payload_attempt"] = payload.copy()
    st.session_state["last_payload_attempt_ts"] = ts
    st.session_state["last_request_id"] = req_id
    # clear previous error until we get a new response
    st.session_state["last_error"] = None

    try:
        result = api_post("/recommend", payload)
        st.session_state.last_result = result
        st.session_state.last_payload = payload.copy()
        st.session_state.last_payload_ts = ts
        st.session_state.last_request_id = req_id
        st.session_state.last_user_input = user_input
        st.session_state["last_error"] = None
    except Exception as e:
        # try to extract response details when available (requests.HTTPError)
        err_info: dict[str, Any] = {"error": str(e)}
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                err_info["status_code"] = resp.status_code
                err_info["response_text"] = resp.text
            except Exception:
                pass
        st.session_state["last_error"] = err_info
        # keep last_payload_attempt present for inspection
        st.error(f"Recommendation failed: {e}")


def _on_title_select():
    # record selection so main flow can handle it after widgets are created
    st.session_state["title_selected"] = st.session_state.get("title_select", "")


# Title select callback removed: we now run recompute immediately after selection.


def _on_global_select(session_key: str, select_key: str):
    """Callback for global selectboxes used to add favorites.
    Appends the selected item to the favorites list and clears the selectbox.
    """
    sel = st.session_state.get(select_key, "")
    if sel:
        if sel not in st.session_state[session_key]:
            st.session_state[session_key].append(sel)
            st.session_state[f"{session_key}_changed"] = True
        # reset the selectbox so it doesn't keep the selection
        st.session_state[select_key] = ""


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
        st.caption(
            f"Effective formula: {sim_w:.2f} × similarity + {rating_w:.2f} × normalized average_rating"
        )
    else:
        sim_w = 1.0
        rating_w = 0.0
        st.caption(
            "Pure cosine similarity for title/profile modes; cold-start modes use average_rating."
        )

    st.divider()
    st.header("Upload books CSV")
    uploaded = st.file_uploader(
        "books_model.csv",
        type=["csv"],
        help=(
            "CSV must contain: record_id, goodreads_book_id, title, authors, average_rating, tags_string."
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

# Load all titles once for client-side selectboxes (search-as-you-type). Stored in session_state to avoid repeated fetches.
if "all_titles" not in st.session_state:
    try:
        limit = int(info.get("books", 10000)) if info.get("books") else 10000
    except Exception:
        limit = 10000
    try:
        st.session_state["all_titles"] = api_get("/titles", q="", limit=limit).get("titles", [])
    except Exception:
        st.session_state["all_titles"] = []

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
        # Single unified title input: type to search, or pick a suggestion from an inline dropdown
        # Keep the typed value in session_state under 'title_input'
        if "title_input" not in st.session_state:
            st.session_state["title_input"] = ""

        # (no pre-processing here; selection handled by on_change callback)

        # Use a single selectbox backed by the full title list (client-side search-as-you-type)
        all_titles = st.session_state.get("all_titles", [])
        options = [""] + all_titles if all_titles else [""]
        # create the selectbox and run recommendation when user picks a value
        st.selectbox("Book title (type to filter)", options=options, index=0, key="title_select", on_change=_on_title_select)
        # determine the currently-selected title (prefer explicit selectbox value)
        selected_value = st.session_state.get("title_select", "") or st.session_state.get("last_title_selected", "")
        # if there was a new selection (recorded via callback), run it once
        if st.session_state.get("title_selected"):
            sel = st.session_state.pop("title_selected")
            if sel and sel != st.session_state.get("last_title_selected", ""):
                st.session_state["last_title_selected"] = sel
                selected_value = sel
                payload.update({"mode": "title", "title": sel})
                user_input = sel
                run_recommendation(payload, user_input)
        # ensure payload contains the last selected title so Generate will send a valid request
        if selected_value:
            payload.update({"mode": "title", "title": selected_value})
            user_input = selected_value
        else:
            payload.update({"mode": "title", "title": ""})
            user_input = ""
    elif mode_label == "Cold start: top-rated books":
        st.info(
            "No user history is needed. The app ranks books by average_rating because the current export has no real ratings_count column."
        )
        payload.update({"mode": "global"})
        user_input = "top-rated"
    elif mode_label == "Cold start: top-rated by tag":
        # empty by default for better UX
        genre = st.text_input("Tag/genre", value="", placeholder="e.g. fantasy")
        payload.update({"mode": "genre", "genre": genre.strip()})
        user_input = genre.strip()
    elif mode_label == "User profile from favorite books":
        # Use a small favorites manager UI so user can build a list interactively
        st.write("Build your list of favorite books (click suggestions to add)")
        favs = favorites_manager(session_key="profile_favs", input_key="profile_fav_input", suggestion_prefix="profile")
        payload.update({"mode": "profile", "favorite_titles": favs})
        user_input = "; ".join(favs)
    else:
        st.write("Smart mode: leave empty for top-rated, type a tag, exact title for similar books, or build a favorites profile")
        auto_text = st.text_input(
            "Smart input (type or pick a suggestion)", value="", placeholder=""
        )

        # favourites manager for smart mode
        favs = favorites_manager(session_key="auto_favs", input_key="auto_fav_input", suggestion_prefix="auto")

        # prefer favorites if any were added, otherwise use typed text
        if favs:
            payload.update({"mode": "auto", "favorite_titles": favs})
            user_input = "; ".join(favs)
            # auto recompute will be handled centrally after inputs are rendered
        else:
            payload.update({"mode": "auto", "title": auto_text.strip()})
            user_input = auto_text.strip()
            # if user typed a title and it matches a suggestion, we could auto-run
            if user_input:
                # do not auto-run on every keystroke; rely on explicit Generate for typed text
                pass

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None

    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

    if st.button("Generate recommendations", type="primary"):
        # Basic front-line validation to avoid creating a request we know will fail.
        m = payload.get("mode")
        if m == "title" and not payload.get("title"):
            st.warning("Please pick a title before requesting recommendations.")
        elif m == "genre" and not payload.get("genre"):
            st.warning("Please enter a genre/tag before requesting recommendations.")
        elif m == "profile" and not payload.get("favorite_titles"):
            st.warning("Please add favorite books to build a profile before requesting recommendations.")
        else:
            run_recommendation(payload, user_input)

    # Centralized auto-run triggers (handle after inputs are displayed):
    # if favorites list changed, session_state contains <session_key>_changed flags

    # profile autos
    if st.session_state.get("profile_favs_changed"):
        st.session_state["profile_favs_changed"] = False
        run_recommendation(payload, user_input)

    # auto_favs (from smart mode)
    if st.session_state.get("auto_favs_changed"):
        st.session_state["auto_favs_changed"] = False
        run_recommendation(payload, user_input)

    if st.session_state.last_result is not None or st.session_state.get("last_error"):
        # Show compact metadata about the last request attempt to aid debugging and log correlation
        with st.expander("Last request details", expanded=False):
            ts = st.session_state.get("last_payload_attempt_ts") or st.session_state.get("last_payload_ts")
            req_id = st.session_state.get("last_request_id")
            st.write(f"Timestamp (UTC): {ts}")
            if req_id:
                st.write(f"Request ID: {req_id}")
            if st.session_state.get("last_payload_attempt"):
                try:
                    st.json(st.session_state.get("last_payload_attempt"))
                except Exception:
                    st.write(st.session_state.get("last_payload_attempt"))
                # offer a download of the payload to ease correlation with backend logs
                try:
                    import json

                    payload_str = json.dumps(st.session_state.get("last_payload_attempt"), indent=2)
                    filename = f"payload_{req_id[:8]}_{(ts or '').replace(':', '-')}.json"
                    st.download_button("Download payload JSON", payload_str, file_name=filename, mime="application/json")
                except Exception:
                    pass
            if st.session_state.get("last_error"):
                st.error(f"Last error: {st.session_state.get('last_error')}")
            # show raw response when available for debugging
            if st.session_state.get("last_result"):
                try:
                    st.subheader("Last raw response")
                    st.json(st.session_state.get("last_result"))
                except Exception:
                    st.write(st.session_state.get("last_result"))

        if st.session_state.last_result is not None:
            render_items(
                st.session_state.last_result["items"],
                st.session_state.last_result["strategy"],
                st.session_state.last_payload["mode"],
                st.session_state.last_user_input,
            )

with tabs[1]:
    st.header("Active deployed model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", info["books"])
    c2.metric("TF‑IDF rows", info["tfidf_shape"][0])
    c3.metric("TF‑IDF features", info["tfidf_shape"][1])
    if info.get("cosine_sim_loaded_from_file"):
        st.success("Precomputed cosine_sim matrix was loaded from file.")
    else:
        build_time = info.get("model_build_time_seconds")
        if build_time:
            st.success(f"Cosine similarity matrix computed at startup in {build_time:.1f}s.")
        else:
            st.info("Cosine similarity matrix will be computed by the API on startup.")
    if info.get("has_ratings_count"):
        st.info(
            "The active CSV contains ratings_count. The UI displays it, but current cold-start ranking still uses average_rating for clarity."
        )
    else:
        st.info(
            "The active CSV has no ratings_count, so cold-start ranking is top-rated by average_rating, not popularity-based."
        )
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
    st.header("Risk assesment")
    st.subheader("Book coverage")
    st.markdown("""
    Out reccomnedation system does not cover all books.
    It can only recommend books that exist in the dataset used by the project (10k goodbooks dataset). 
    If a book is not included in the dataset, the recommender cannot suggest it or compare it properly.
    Users might come expecting the recommendation system to include all existing books, however, this is not the case.
    
    Risk level: Medium
    """)
    st.subheader("Reccomendation quality")
    st.markdown("""
    Content-based filtering recommends similar books based on metadata, which is useful, especially for the cold-start problem.
    However, since we are not comparing the similarity of taste across different users, like collaborative filtering, the personalization of reccomendations is limited.
    The system does not truly “understand” books and users. It compares text/features mathematically using a TF-IDF and cosine similarity.
    Due to this, some recommendations may not be as accurate since the system relies only on the similarity of books using tags.
                
    Risk level: Low-Medium
    """)
    st.subheader("Dataset bias")
    st.markdown("""
    The recommendation system reflects the dataset it is built on.
    If the dataset contains mostly popular, English-language, or highly rated books, the recommendations may also favor these types of books. This can make newer books, niche books, non-English books, or less popular authors less visible.
    Because of this, the recommendations should not be considered fully neutral or representative of all books.

    Risk level: Medium
    """)
    st.subheader("Performance and scalability")
    st.markdown("""
    The recommendation system may become slower if it has to compare many books at request time.
    TF-IDF and cosine similarity work well for a dataset of this size, but performance could become an issue if the dataset grows significantly (for example to millions of books) or if many users access the application at the same time.
    
    Risk level: Medium
    """)
    st.subheader("Deployment")
    st.markdown("""
    The project is generally safe to deploy as a basic reccomender, since it does not use any personal data and does not create users.
    The privacy risk is relatively low. The users can simply input the books they have liked and the appliacation outputs similar books they could like.

    Risk level: Low-Medium
    """)
    st.subheader("User feedback limitation")
    st.markdown("""
    The application allows users to mark each recommendation as useful or not useful.
    This feedback is saved into a CSV file and can be used later for analysis, reranking, or personalization. However, in the current version, the feedback is not yet used to change future recommendations automatically.
    Because of this, the feedback system is useful for collecting evaluation data, but it does not currently improve the recommendation model in real time.

    **Risk level:** Low-Medium
    """)

with tabs[3]:
    st.header("Installation manual")
    st.markdown(
        ""
        "1. Docker desktop is installed and running"
        "2. Open terminal in the project folder and run:"
        ""
    )
    st.code("docker compose up --build", language="bash")
    st.write("Open GUI at http://localhost:8501 and API docs at http://localhost:8000/docs.")
    st.header("User manual")
    st.markdown(
        """

## Goal

The application recommends books using content-based filtering from Goodbooks tag profiles.

## Recommendation modes

1. **Similar books by title** — choose one known book and receive books with similar cleaned tags.
2. **Cold start: top-rated books** — for users with no preferences yet; ranks by `average_rating`.
3. **Cold start: top-rated by tag** — enter a tag/genre such as `fantasy`, `dystopia`, or `mystery`; results are ranked by `average_rating`.
4. **User profile from favorite books** — select several favorite books; the system builds an average TF-IDF user vector.
5. **Smart mode** — automatically chooses the strategy based on the input.

## Options

- `Number of recommendations`: top-N result size.
- `Use average-rating reranking`: for similarity-based modes, combines content similarity with normalized rating.
- `Content similarity weight`: controls the reranking formula. Rating weight is automatically computed as `1 - content_weight`.

## Feedback

Open a recommendation card and click **Useful**. The app stores this feedback for future improvements.
        """
    )
