# Installation Manual

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose

## Required runtime artifacts

The project includes `data/books_model.csv`. The cosine similarity matrix is
computed automatically at container startup from the active CSV; no precomputed
matrix file is required.

## Start

```bash
docker compose down
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
