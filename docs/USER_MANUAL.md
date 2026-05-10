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

Open a recommendation card and click **Useful** or **Not useful**. The app stores this feedback for future improvements.
