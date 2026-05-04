# User Manual

## Goal

The application recommends books using content-based filtering from Goodbooks tag profiles.

## Recommendation modes

1. **Similar books by title** — choose one known book and receive similar books.
2. **Cold start: global popularity** — for users with no preferences yet.
3. **Cold start: genre** — enter a tag/genre such as `fantasy`, `dystopia`, or `mystery`.
4. **User profile** — select several favorite books; the system builds an average TF-IDF user vector.
5. **Auto wrapper** — automatically chooses the strategy based on the input.

## Options

- `Number of recommendations`: top-N result size.
- `Use hybrid score`: combine content similarity with normalized rating.
- `Similarity weight` and `Rating weight`: tune ranking behavior.

## Feedback

Open a recommendation card and click **Useful** or **Not useful**. The app stores this feedback for future improvements.
