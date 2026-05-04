# Quality Evaluation and Risk Assessment

## Quality evaluation

Mini-project 1 evaluated the recommender with leave-one-out testing:

1. Pick a random sample of users.
2. For each user, books with rating >= 4 are considered relevant.
3. One relevant book is used as the query.
4. Remaining relevant books are ground truth.
5. Compute Precision@10, Recall@10, and F1@10.

Best deployed experiment:

| Experiment | Precision@10 | Recall@10 | F1@10 | Users |
|---|---:|---:|---:|---:|
| exp5_maxfeat | 0.0806 | 0.0957 | 0.0875 | 428 |

These values are meaningful because the dataset has 10,000 books and random top-10 matching would be around 1%.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cold start | No user history | Global and genre popularity fallbacks |
| Noisy tags | Wrong similarity | MP1 preprocessing removes filler tags such as `to-read`, `favorites`, `owned` |
| Popularity bias | Popular books dominate | Adjustable hybrid-score weights |
| Memory usage | Full 10k×10k cosine matrix can be large | Load once in API; future improvement: approximate nearest neighbors |
| Static model | User feedback not learned automatically | Feedback collection added for future reranking |
| Ambiguous titles | Duplicate titles may exist | First matching title is used, same as MP1 notebook logic |

## Improvement proposal

Add a feedback-based reranker that learns from useful/not-useful clicks, personalizes tag weights, and monitors performance drift over time.
