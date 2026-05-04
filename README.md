# Mini-project 3: ISA Deployment — Goodbooks Content-Based Recommender

This is a production-ready Docker deployment of **Mini-project 1**, based on the uploaded notebooks:

- `projekt1_EDA.ipynb`
- `projekt1_preprocessing.ipynb`
- `projekt1_recsys.ipynb`

The deployed model is **not collaborative filtering**. It is the actual Mini-project 1 approach:

> cleaned Goodreads book tags → `tags_string` profile → TF-IDF vectors → cosine similarity → top-N recommendations

The best MP1 experiment was `exp5_maxfeat`, using:

```python
TfidfVectorizer(stop_words="english", max_features=5000)
```

with the benchmark result:

| Experiment | Precision@10 | Recall@10 | F1@10 | Users |
|---|---:|---:|---:|---:|
| exp5_maxfeat | 0.0806 | 0.0957 | 0.0875 | 428 |

## Assignment coverage

### 3.1 Intelligent System Application

- **A — UI/GUI/Web GUI:** Streamlit GUI on port `8501`
- **B — Deployment of RecSys model:** FastAPI service on port `8000`, deploying the MP1 content-based recommender
- **C — Quality evaluation and risk assessment:** evaluation dashboard + risk assessment tab

### 3.2 Production-ready package and documentation

- **A — Docker image:** `docker compose up --build` builds API and GUI containers
- **B — Installation manual and user manual:** included below and in the GUI

## Run

```bash
unzip miniproject3_content_based_goodbooks.zip
cd miniproject3_content_based_goodbooks
docker compose up --build
```

Open:

- GUI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

## Use your full Mini-project 1 model/data

The package is runnable out of the box with a small sample `books_model.csv`, because the raw Goodbooks dataset was not uploaded with the notebooks.

For final submission, replace or upload the full file produced in `projekt1_recsys.ipynb`:

```text
models/books_model.csv
```

Expected columns:

```text
record_id, goodreads_book_id, title, authors, average_rating, tags_string
```

Optional but recommended:

```text
ratings_count, popularity_score
```

Ways to activate the full file:

1. In the GUI sidebar, upload `books_model.csv`.
2. Or replace `data/sample/books_model.csv` before running Docker.
3. Or set `BOOKS_MODEL_PATH` in `docker-compose.yml`.

## Recommendation modes

### 1. Similar books by title

Uses TF-IDF/cosine similarity over tag profiles. Example:

```text
Harry Potter and the Sorcerer's Stone → similar fantasy/magic/school books
```

### 2. Cold start: global popularity

Used when a new user gives no input. Sorts by `popularity_score`.

### 3. Cold start: genre popularity

User enters a genre/tag, e.g. `fantasy`, `dystopia`, `mystery`.

### 4. User profile recommender

User selects favorite books. The app averages their TF-IDF vectors and recommends books closest to the average profile.

### 5. Auto wrapper

Implements the wrapper idea from the notebook:

- empty input → global popularity
- short string → genre popularity
- exact title → content-based recommendations
- list of books → user-profile recommender

## Hybrid score

The app supports the MP1 hybrid scoring idea:

```text
hybrid_score = 0.8 × similarity + 0.2 × normalized average_rating
```

The GUI lets the user change these weights interactively.

## Feedback system

Each recommendation can be marked as useful or not useful. Feedback is stored in:

```text
data/sample/feedback.csv
```

This satisfies the risk-assessment improvement proposal: feedback can later be used for reranking or personalization.

## Project structure

```text
miniproject3_content_based_goodbooks/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── recommender.py
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── streamlit_app.py
├── data/sample/
│   ├── books_model.csv
│   └── feedback.csv
├── results/
│   └── experiment_results.csv
├── scripts/
│   └── build_from_goodbooks.py
├── docs/
│   ├── INSTALLATION_MANUAL.md
│   ├── USER_MANUAL.md
│   └── RISK_ASSESSMENT.md
└── docker-compose.yml
```
