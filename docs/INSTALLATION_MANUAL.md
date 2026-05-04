# Installation Manual

## Requirements

- Docker
- Docker Compose

## Steps

```bash
unzip miniproject3_content_based_goodbooks.zip
cd miniproject3_content_based_goodbooks
docker compose up --build
```

## Open the app

- Streamlit GUI: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs

## Stop the app

```bash
docker compose down
```

## Use full Mini-project 1 output

Upload your full `books_model.csv` in the GUI sidebar or replace:

```text
data/sample/books_model.csv
```

The API rebuilds the TF-IDF matrix and cosine similarity matrix after upload.
