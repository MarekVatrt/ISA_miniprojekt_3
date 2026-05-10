# Mini-project 3 Documentation

## 1. Goal

The goal is to deploy the Goodbooks recommender developed in Mini-project 1 as an Intelligent System Application. The delivered application contains a GUI, model inference API, evaluation dashboard, risk assessment, Docker deployment, installation manual, and user manual.

## 2. Application architecture

The active deployment has two services:

1. **FastAPI backend** – loads the Goodbooks content-based recommender, exposes endpoints for recommendations, model information, experiment metrics, and feedback collection.
2. **Streamlit GUI** – provides an interactive web interface for recommendations, model inspection, risk assessment, and manuals.

The services are orchestrated by Docker Compose.

## 3. Data and model artifacts

The backend expects:

```text
data/books_model.csv
results/experiment_results.csv
```

Required `books_model.csv` columns:

```text
record_id, goodreads_book_id, title, authors, average_rating, tags_string
```

The cosine similarity matrix is generated at API startup from the active CSV so
no precomputed matrix file is required.

## 4. Recommender model

The active model is content-based filtering:

```text
cleaned tags_string
→ TF-IDF vectors, max_features=5000
→ cosine similarity between books
```

For title-based recommendations, the selected book is used as the query and the system returns the most similar books. For user-profile recommendations, selected favorite books are averaged into one TF-IDF profile vector.

For similarity-based modes, the app can rerank results by combining cosine similarity with normalized `average_rating`:

```text
hybrid_score = content_weight × cosine_similarity + rating_weight × normalized_average_rating
```

The default is `0.8 × similarity + 0.2 × normalized average_rating`.

## 5. Cold-start strategy

The current full export has `average_rating`, but not a reliable `ratings_count`, so the application does not claim true popularity. Cold-start modes are labelled as:

- **top-rated books**
- **top-rated books by tag**

Both are ranked by `average_rating`.

## 6. GUI functionality

The GUI supports:

- title-based similar-book recommendations;
- top-rated cold start;
- tag-based top-rated cold start;
- user-profile recommendations from favorite books;
- smart mode that selects a strategy automatically;
- adjustable average-rating reranking for similarity modes;
- recommendation feedback;
- model/evaluation dashboard;
- risk assessment and manuals.

## 7. Quality evaluation

The app displays the Mini-project 1 experiment results:

- Precision@10;
- Recall@10;
- F1@10.

The selected deployment experiment is `exp5_maxfeat`, using `TfidfVectorizer(stop_words='english', max_features=5000)` on `tags_string`.

## 8. Risk assessment

Main risks and mitigations:

| Risk | Impact | Mitigation |
|---|---|---|
| Cold-start users | No personalization | top-rated and tag-based fallback |
| No `ratings_count` in current export | no true popularity estimate | label as top-rated; future export can add count |
| Noisy tags | bad similarity | preprocessing removes filler tags |
| Large cosine matrix | memory cost | load once with `mmap_mode`; future top-N export |
| Feedback not used yet | no online learning | feedback is stored for future reranking |

## 9. Installation manual

```bash
docker compose up --build
```

Open:

- GUI: `http://localhost:8501`
- API: `http://localhost:8000/docs`

Stop:

```bash
docker compose down
```

## 10. User manual

1. Open the GUI.
2. Choose a recommendation mode.
3. Select a book, tag, or favorite books.
4. Adjust the number of recommendations.
5. Optionally enable average-rating reranking.
6. Click **Generate recommendations**.
7. Inspect model status and risks in the other tabs.
