from __future__ import annotations

import os
import json
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

#nacitaj API URL z env, fallback na localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

#konfiguracia streamlit appky
st.set_page_config(
    page_title="Mini-project 3 | Goodbooks Recommender",
    page_icon="📚",
    layout="wide",
)

#API helper funkcie pre GET a POST requesty 
def api_get(path: str, **params) -> dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# UI komponenty pre spravu oblubenych knih a zobrazenia odporucani
def favorites_manager(session_key="favorites_list", suggestion_prefix="fav"):
    """ 
    Interakticny UI komponent na budovanie zoznamu oblubenych knih

    session_key: kluc pre streamlit session state, kde sa bude uchovavat aktualny zoznam oblubenych knih
    suggestion_prefix: prefix pre unikatne kluce UI komponentov

    funkcia vracia list oblubenych titulov
    """

    #vytvorit prazdny list v session state, ak neexistuje
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    #kontrola ci je list zmeneny
    changed_flag = f"{session_key}_changed"
    if changed_flag not in st.session_state:
        st.session_state[changed_flag] = False

    #dropdown s navrhmi titulov
    all_titles = st.session_state.get("all_titles", [])
    options = [""] + all_titles if all_titles else [""]
    sel = st.selectbox("Add favorite (type to filter and pick)", options=options, index=0, key=f"{suggestion_prefix}_global_select")
    if sel:
        if sel not in st.session_state[session_key]:
            st.session_state[session_key].append(sel)
            st.session_state[changed_flag] = True

    #ukazat aktualny zoznam knih s moznostou vymazania
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
    """ 
    Zobrazenie odporucani v UI spolu s relevantnymi informaciami a feedback tlacidlami

    items: list slovnikov s informaciami o odporucanych knihach
    strategy: popis pouzitej strategie odporucania
    mode: pouzity rezim odporucania (title/genre/profile/global)
    user_input: vstup od pouzivatela
    """
    
    st.subheader(strategy)

    #handle prazdne vysledky
    if not items:
        st.warning(
            "No recommendations found. Try another title/tag, or check that the active books_model.csv is compatible with the cosine matrix."
        )
        return

    #priprav a zobraz tabulku vysledkov
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

    # zobrazit vizualizacie vedla seba ak su k dispozicii
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

    #zobrazit detailne informacie pre kazde odporucanie s tlacidlom pre feedback
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
            
            #feedback tlacidla
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

#hlavna logika pre odporucania
def run_recommendation(payload: dict, user_input: str) -> None:
    """ 
    Vykonaj request na odporucenie a updateuj session state s vysledkom.
    Obsahuje klientsku validaciu vstupov a spravu chyb.

    payload: dict s parametrami pre API request
    user_input: vstup od pouzivatela, pre feedback a logovanie
    """

    # validacia vstupov pre prevenciu zbytocnych requestov
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

    #generovanie metadat pre request correlation a debugging
    ts = datetime.utcnow().isoformat() + "Z"
    req_id = uuid.uuid4().hex

    #ulozenie metadat pokusu
    st.session_state["last_payload_attempt"] = payload.copy()
    st.session_state["last_payload_attempt_ts"] = ts
    st.session_state["last_request_id"] = req_id
    st.session_state["last_error"] = None

    try:
        # volaj API a uloz uspesny vysledok 
        result = api_post("/recommend", payload)
        st.session_state.last_result = result
        st.session_state.last_payload = payload.copy()
        st.session_state.last_payload_ts = ts
        st.session_state.last_request_id = req_id
        st.session_state.last_user_input = user_input
        st.session_state["last_error"] = None
    except Exception as e:
        # zachyt error a uloz informacie pre debugging
        err_info: dict[str, Any] = {"error": str(e)}
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                err_info["status_code"] = resp.status_code
                err_info["response_text"] = resp.text
            except Exception:
                pass
        st.session_state["last_error"] = err_info
        st.error(f"Recommendation failed: {e}")


# callbacky
def _on_title_select():
    st.session_state["title_selected"] = st.session_state.get("title_select", "")


# hlavny layout aplikacie
st.title("📚 Goodbooks Recommender")
st.caption(
    "Active model: cleaned book tag profiles → TF‑IDF max_features=5000 → cosine similarity. "
    "For similarity-based modes, recommendations can be reranked with average_rating."
)

# sidebar pre model controls a upload noveho CSV suboru, ktory sa pouzije pre model
with st.sidebar:
    st.header("Model controls")
    # pocet odporucani, hybrid reranking a jeho vahy
    n = st.slider("Number of recommendations", 1, 30, 10)
    hybrid = st.toggle(
        "Use average-rating reranking",
        value=True,
        help="For similarity-based modes, blend cosine similarity with normalized average_rating. Cold-start modes are already top-rated by average_rating.",
    )

    # vahovy slider (zobrazeny len pri hybridnom rezime)
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

    # CSV nacitanie pre vlastny model
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

    #reset modela na defaultny stav
    if st.button("Reset configured model"):
        api_post("/reset-sample-model", {})
        st.success("Configured model restored.")

#kontrola API a nacitanie informacii o modeli
try:
    info = api_get("/model-info")
except Exception as e:
    st.error(
        f"API is not available at {API_URL}. Start the app with docker compose up --build. Error: {e}"
    )
    st.stop()

# nacitanie vsetkych kniznych titulov pre dropdown - search-as-you-type
if "all_titles" not in st.session_state:
    try:
        limit = int(info.get("books", 10000)) if info.get("books") else 10000
    except Exception:
        limit = 10000
    try:
        st.session_state["all_titles"] = api_get("/titles", q="", limit=limit).get("titles", [])
    except Exception:
        st.session_state["all_titles"] = []

# tab rozlozenie
tabs = st.tabs(["Recommend", "Model & Evaluation", "Risk Assessment", "Manuals"])

# tab 1 - hlavna funkcionalita odporucania
with tabs[0]:
    # vyber modu odporucania a zobrazenie vstupov pre dany mod
    mode_label = st.radio(
        "Recommendation mode",
        [
            "Similar books by title",
            "Cold start: top-rated books",
            "Cold start: top-rated by tag",
            "User profile from favorite books",
        ],
        horizontal=True,
    )
    # popis rezimu a relevantnych vstupov
    payload = {
        "n": n,
        "hybrid": hybrid,
        "similarity_weight": sim_w,
        "rating_weight": rating_w,
    }
    user_input = ""

    # iputy zavisia dynamicky od vstupu/vybraneho rezimu, ale vzdy sa ukladaju do payloadu pre API request a user_input pre feedback/logging
    if mode_label == "Similar books by title":
        all_titles = st.session_state.get("all_titles", [])
        options = [""] + all_titles if all_titles else [""]
        st.selectbox("Book title (type to filter)", options=options, index=0, key="title_select", on_change=_on_title_select)
        
        # vykonava title selection a auto-trigger recommendation pri zmene vyberu
        selected_value = st.session_state.get("title_select", "") or st.session_state.get("last_title_selected", "")
        if st.session_state.get("title_selected"):
            sel = st.session_state.pop("title_selected")
            if sel and sel != st.session_state.get("last_title_selected", ""):
                st.session_state["last_title_selected"] = sel
                selected_value = sel
                payload.update({"mode": "title", "title": sel})
                user_input = sel
                run_recommendation(payload, user_input)

        # update payloadu pri aktualnom vybere        
        if selected_value:
            payload.update({"mode": "title", "title": selected_value})
            user_input = selected_value
        else:
            payload.update({"mode": "title", "title": ""})
            user_input = ""
    
    elif mode_label == "Cold start: top-rated books":
        # tento rezim nevyzaduje zadanie vstupu od pouzivatela
        st.info(
            "No user history is needed. The app ranks books by average_rating because the current export has no real ratings_count column."
        )
        payload.update({"mode": "global"})
        user_input = "top-rated"
    
    elif mode_label == "Cold start: top-rated by tag":
        # textovy input pre tag/zaner, ktory sa pouzije pre hladanie v modeli a update payloadu
        genre = st.text_input("Tag/genre", value="", placeholder="e.g. fantasy")
        payload.update({"mode": "genre", "genre": genre.strip()})
        user_input = genre.strip()
    
    elif mode_label == "User profile from favorite books":
        # interaktivny komponent pre budovanie zoznamu oblubenych knih, ktory sa pouzije pre profilovy rezim
        st.write("Build your list of favorite books (click suggestions to add)")
        favs = favorites_manager(session_key="profile_favs", suggestion_prefix="profile")
        payload.update({"mode": "profile", "favorite_titles": favs})
        user_input = "; ".join(favs)

    # inicializacia session state pre spravu posledneho requestu a vysledku

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None

    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

    # manualne spustenie odporucania tlacidlom

    if st.button("Generate recommendations", type="primary"):
        m = payload.get("mode")
        if m == "title" and not payload.get("title"):
            st.warning("Please pick a title before requesting recommendations.")
        elif m == "genre" and not payload.get("genre"):
            st.warning("Please enter a genre/tag before requesting recommendations.")
        elif m == "profile" and not payload.get("favorite_titles"):
            st.warning("Please add favorite books to build a profile before requesting recommendations.")
        else:
            run_recommendation(payload, user_input)

    # auto-trigger odporucania pri zmeene oblubenych knih v profile rezime

    if st.session_state.get("profile_favs_changed"):
        st.session_state["profile_favs_changed"] = False
        run_recommendation(payload, user_input)

    # zobrazenie vysledkov a informacii pre debuggovanie
    if st.session_state.last_result is not None or st.session_state.get("last_error"):
        # zobrazit expander s detailami posledneho requestu
        with st.expander("Last request details", expanded=False):
            ts = st.session_state.get("last_payload_attempt_ts") or st.session_state.get("last_payload_ts")
            req_id = st.session_state.get("last_request_id")
            st.write(f"Timestamp (UTC): {ts}")
            if req_id:
                st.write(f"Request ID: {req_id}")
            
            # zobrazenie payloadu JSON
            if st.session_state.get("last_payload_attempt"):
                try:
                    st.json(st.session_state.get("last_payload_attempt"))
                except Exception:
                    st.write(st.session_state.get("last_payload_attempt"))
                
                # tlacidlo na stiahnutie payloadu ako JSON
                try:
                    payload_str = json.dumps(st.session_state.get("last_payload_attempt"), indent=2)
                    filename = f"payload_{req_id[:8]}_{(ts or '').replace(':', '-')}.json"
                    st.download_button("Download payload JSON", payload_str, file_name=filename, mime="application/json")
                except Exception:
                    pass
            
            # zobraz chybu ak existuje
            if st.session_state.get("last_error"):
                st.error(f"Last error: {st.session_state.get('last_error')}")
            # zobraz RAW API response
            if st.session_state.get("last_result"):
                try:
                    st.subheader("Last raw response")
                    st.json(st.session_state.get("last_result"))
                except Exception:
                    st.write(st.session_state.get("last_result"))

        # renderovanie odporucani ak je vysledok k dispozicii
        if st.session_state.last_result is not None:
            render_items(
                st.session_state.last_result["items"],
                st.session_state.last_result["strategy"],
                st.session_state.last_payload["mode"],
                st.session_state.last_user_input,
            )

# tab for model evaluation a benchmark z mini-project 1 notebooku
with tabs[1]:
    st.header("Active deployed model")

    # ukaz metriky modelu a informacie o nacitani/casoch buildovania
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", info["books"])
    c2.metric("TF‑IDF rows", info["tfidf_shape"][0])
    c3.metric("TF‑IDF features", info["tfidf_shape"][1])
    
    # ukaz info o nacitani modelu
    if info.get("cosine_sim_loaded_from_file"):
        st.success("Precomputed cosine_sim matrix was loaded from file.")
    else:
        build_time = info.get("model_build_time_seconds")
        if build_time:
            st.success(f"Cosine similarity matrix computed at startup in {build_time:.1f}s.")
        else:
            st.info("Cosine similarity matrix will be computed by the API on startup.")
    
    # zobrazit ratings_count info
    if info.get("has_ratings_count"):
        st.info(
            "The active CSV contains ratings_count. The UI displays it, but current cold-start ranking still uses average_rating for clarity."
        )
    else:
        st.info(
            "The active CSV has no ratings_count, so cold-start ranking is top-rated by average_rating, not popularity-based."
        )
    # zobrazenie full model info
    st.json({k: v for k, v in info.items() if k not in {"best_experiment"}})

    # zobrazenie benchmark vysledkov z miniprojekt 1
    st.subheader("Benchmark from Mini-project 1 notebook")
    experiments = pd.DataFrame(api_get("/experiments")["experiments"])
    
    if not experiments.empty:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
        # Bar plot pre porovnanie experimentov
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
    
    # evluacia metodiky hodnotenia a diskusia o vysledkoch
    st.markdown(
        """
        **Quality evaluation approach:** leave-one-out on sampled users. For each user,
        ratings ≥ 4 are relevant books; one is used as query and the rest are ground truth.
        Metrics: Precision@K, Recall@K, F1@K.
        """
    )

# tab pre risk asessment a diskusiu o obmedzeniach modelu a aplikacie
with tabs[2]:
    st.header("Risk Assessment")
   
    st.subheader("Book coverage")
    st.markdown("""
    Our recommendation system does not cover all books.
    It can only recommend books that exist in the dataset used by the project (10k goodbooks dataset). 
    If a book is not included in the dataset, the recommender cannot suggest it or compare it properly.
    Users might come expecting the recommendation system to include all existing books, however, this is not the case.
    
    Risk level: Medium
    """)
    
    st.subheader("Recommendation quality")
    st.markdown("""
    Content-based filtering recommends similar books based on metadata, which is useful, especially for the cold-start problem.
    However, since we are not comparing the similarity of taste across different users, like collaborative filtering, the personalization of recommendations is limited.
    The system does not truly "understand" books and users. It compares text/features mathematically using a TF-IDF and cosine similarity.
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
    The project is generally safe to deploy as a basic recommender, since it does not use any personal data and does not create users.
    The privacy risk is relatively low. The users can simply input the books they have liked and the application outputs similar books they could like.

    Risk level: Low-Medium
    """)
    
    st.subheader("User feedback limitation")
    st.markdown("""
    The application allows users to mark each recommendation as useful or not useful.
    This feedback is saved into a CSV file and can be used later for analysis, reranking, or personalization. However, in the current version, the feedback is not yet used to change future recommendations automatically.
    Because of this, the feedback system is useful for collecting evaluation data, but it does not currently improve the recommendation model in real time.

    **Risk level:** Low-Medium
    """)

# manualy a instrukcie pre pouzivanie aplikacie a deploymentu
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

## Options

- `Number of recommendations`: top-N result size.
- `Use average-rating reranking`: for similarity-based modes, combines content similarity with normalized rating.
- `Content similarity weight`: controls the reranking formula. Rating weight is automatically computed as `1 - content_weight`.

## Feedback

Open a recommendation card and click **Useful**. The app stores this feedback for future improvements.
        """
    )