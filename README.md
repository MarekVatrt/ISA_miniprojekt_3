# Installation Manual

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose
- Full miniprojekt3 folder

## Required runtime artifacts

The project includes `data/books_model.csv`. The cosine similarity matrix is
computed automatically at container startup from the active CSV (no precomputed
matrix file is required). books_model can be replaced by another csv file, however, it must contain the required columns. 

## Start

```bash
docker compose up --build
```

Open:

- GUI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Model info: `http://localhost:8000/model-info`

## Verify model loading

```bash
curl http://localhost:8000/model-info
```

The `/model-info` endpoint shows whether the model is ready and provides
details such as TF-IDF shape and number of books.

## Stop

```bash
docker compose down
```

# User Manual

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
