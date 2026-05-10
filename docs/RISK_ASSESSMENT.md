# Quality Evaluation and Risk Assessment

## Quality evaluation

Mini-project 1 evaluated the recommender with leave-one-out testing:

1. Pick a random sample of users.
2. For each user, books with rating `>= 4` are considered relevant.
3. One relevant book is used as the query.
4. Remaining relevant books are ground truth.
5. Compute Precision@10, Recall@10, and F1@10.

Best deployed experiment:

| Experiment | Precision@10 | Recall@10 | F1@10 | Users |
|---|---:|---:|---:|---:|
| exp5_maxfeat | 0.0806 | 0.0957 | 0.0875 | 428 |

These values are meaningful because the catalog has 10,000 books and random top-10 matching would be very low.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cold-start user has no history | No personalized vector | Top-rated fallback and tag-based top-rated fallback |
| Current full export has no real `ratings_count` | True popularity/confidence cannot be measured | The app labels fallback as **top-rated**, not popularity-based |
| Rating-only cold start can over-promote niche books | High average rating may come from few ratings | Future export should add `ratings_count` from the original Goodbooks `books.csv` |
| Noisy Goodreads tags | Wrong similarity | MP1 preprocessing removes filler tags such as `to-read`, `favorites`, `owned` |
| Hybrid-score confusion | Users may misunderstand two independent weights | UI uses one content-weight slider; backend normalizes weights |
| Memory usage | Full 10k×10k cosine matrix is large | Load once with `mmap_mode`; future improvement: top-N export or FAISS |
| Static model | Feedback is not learned automatically | Feedback collection is added for future reranking |
| Ambiguous/duplicate titles | Wrong book may be selected | First matching title is used consistently; future improvement: display IDs |

## Improvement proposal

Add `ratings_count` to the export, add a feedback-based reranker that learns from useful/not-useful clicks, personalize tag weights, and monitor performance drift over time.
