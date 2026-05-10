# Mini-project 3: ISA Deployment — Goodbooks Content-Based Recommender

This project deploys the Mini-project 1 Goodbooks recommender as a small production-style application with:

- FastAPI backend on port `8000`
- Streamlit GUI on port `8501`
- Docker Compose deployment
- model/evaluation/risk-assessment pages
- feedback collection for future reranking

The active deployed model is **not the old collaborative-filtering template**. The running Docker app uses:

```text
cleaned Goodreads tag profile (`tags_string`)
→ TF-IDF vectors with max_features=5000
→ cosine similarity between books
→ optional reranking with normalized average_rating
```

## Active model artifacts

For the real 10k-book run, place these files before starting Docker:

```text
data/books_model.csv
models/cosine_sim_best.npy
results/experiment_results.csv
```

Required `books_model.csv` columns:

```text
record_id, goodreads_book_id, title, authors, average_rating, tags_string
```

The current full export contains `average_rating` but not a reliable `ratings_count`, so the app labels cold-start results as **top-rated**, not popularity-based.

The cosine matrix must be generated from the same CSV in the same row order:

```text
cosine_sim_best.npy.shape == (len(books_model.csv), len(books_model.csv))
```

Check after startup:

```bash
curl http://localhost:8000/model-info
```

Expected value when the precomputed matrix is loaded:

```json
"cosine_sim_loaded_from_file": true
```

## Run

```bash
docker compose down
docker compose up --build
```

Open:

- GUI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Large model files are excluded from Docker build context by `.dockerignore` and mounted at runtime through Docker Compose volumes.

## Recommendation modes

### 1. Similar books by title

Uses the deployed TF-IDF/cosine-similarity model. The user selects a book and receives similar books based on cleaned tags.

### 2. Cold start: top-rated books

Used when no preference is available. Since the current export does not include a real `ratings_count`, this mode ranks books by `average_rating` and is intentionally not called popularity.

### 3. Cold start: top-rated by tag

Filters books by a tag/genre such as `fantasy`, `dystopia`, or `mystery`, then ranks the filtered results by `average_rating`.

### 4. User profile from favorite books

Averages the TF-IDF vectors of selected favorite books and recommends books closest to that average profile.

### 5. Smart mode

A decision wrapper over the same strategies:

- empty input → top-rated books
- short text → top-rated books by tag
- exact title → content-based recommendations
- list of favorite books → user-profile recommender

## Average-rating reranking

For similarity-based modes, the app can rerank candidates using:

```text
hybrid_score = content_weight × cosine_similarity + rating_weight × normalized_average_rating
```

Default:

```text
0.8 × similarity + 0.2 × normalized average_rating
```

The UI uses one slider for `content_weight`; `rating_weight` is computed as `1 - content_weight`. The backend also normalizes weights, so edge cases such as `1.0 + 1.0` become `0.5 + 0.5` instead of producing invalid percentages.

## Evaluation

The best displayed Mini-project 1 experiment is:

| Experiment | Precision@10 | Recall@10 | F1@10 | Users |
|---|---:|---:|---:|---:|
| exp5_maxfeat | 0.0806 | 0.0957 | 0.0875 | 428 |

Evaluation approach: leave-one-out on sampled users. Ratings `>= 4` are relevant; one relevant book is used as the query and the remaining relevant books are ground truth.

## Project status

This version is presentation-ready for the first deployed model:

```text
TF-IDF + cosine similarity + optional average-rating reranking
```

Legacy generated files from the earlier collaborative-filtering template are left in the repository for now, but the active Docker deployment uses only:

```text
backend/app/main.py
backend/app/recommender.py
frontend/streamlit_app.py
docker-compose.yml
data/books_model.csv
models/cosine_sim_best.npy
results/experiment_results.csv
```
