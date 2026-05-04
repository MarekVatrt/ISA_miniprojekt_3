# Mini-project 3 Documentation

## 1. Goal

The goal is to deploy a recommender model developed in Mini-project 1 or Mini-project 2 as an Intelligent System Application. The delivered application contains a GUI, model inference API, evaluation dashboard, risk assessment, Docker deployment, installation manual, and user manual.

## 2. Application architecture

The application has two services:

1. **FastAPI backend** – loads or trains the recommendation model, exposes endpoints for recommendations, evaluation metrics, risk assessment, feedback collection, and retraining.
2. **Streamlit GUI** – provides an interactive web interface for users and presentation during the lab session.

The services are orchestrated by Docker Compose. The API must become healthy before the GUI starts.

## 3. Data pipeline

The data pipeline expects interaction data with these required columns:

- `user_id`
- `item_id`
- `rating`

The pipeline performs:

- column normalization and alias handling;
- removal of rows with missing user/item/rating values;
- conversion of ratings to numeric values;
- duplicate user-item aggregation using mean rating;
- construction of a user-item matrix;
- item-item similarity calculation;
- model evaluation and artifact persistence.

## 4. Recommender model

The included deployable model is item-based collaborative filtering. It computes item similarity from centered user ratings and predicts a rating for each candidate item using the weighted ratings of similar items previously rated by the selected user. For cold-start situations, it falls back to item mean, user mean, or global mean.

## 5. GUI functionality

The GUI supports:

- selecting a user;
- choosing the number of recommendations;
- including/excluding already seen items;
- visualizing ranked recommendations;
- uploading a new CSV dataset and retraining/deploying the model;
- saving explicit feedback;
- showing quality metrics;
- showing risk assessment and improvement proposal;
- showing installation and user manual directly in the app.

## 6. Quality evaluation

The app reports:

- **RMSE** – rating prediction error;
- **MAE** – average absolute prediction error;
- **Precision@10** – share of recommended items matching high-rated items;
- **Catalog coverage** – share of catalog appearing in recommendations;
- **Popularity bias** – ratio between the most popular item and average item popularity.

## 7. Risk assessment

Main risks and mitigations:

| Risk | Impact | Mitigation |
|---|---|---|
| Cold-start users/items | Weak recommendations for new users/items | onboarding preferences, metadata fallback, popularity fallback |
| Popularity bias | Over-recommends popular items | diversity-aware ranking, long-tail monitoring |
| Privacy | Interaction logs reveal preferences | pseudonymized IDs, access control, retention policy |
| Model drift | Quality degrades over time | scheduled retraining and metric monitoring |
| Explainability | Users may not trust results | show predicted rating, metadata, popularity, and reasons |

## 8. Proposed improvement

The current app already stores feedback. A production extension should merge feedback into the training dataset, schedule retraining, and compare model variants with A/B tests.

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
2. Select a user.
3. Choose recommendation count.
4. Click **Get recommendations**.
5. Review recommendation table and chart.
6. Save feedback if needed.
7. Present quality metrics and risk assessment from their tabs.
