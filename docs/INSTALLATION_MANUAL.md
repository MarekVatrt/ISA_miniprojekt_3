# Installation Manual

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose

## Required runtime artifacts

The project includes `data/books_model.csv`. For the fastest presentation run, also place the precomputed matrix here:

```text
models/cosine_sim_best.npy
```

The matrix is not included in the ZIP because it is large. It must match the row order of `data/books_model.csv`.

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

Expected when `models/cosine_sim_best.npy` is present and compatible:

```json
"cosine_sim_loaded_from_file": true
```

## Stop

```bash
docker compose down
```
