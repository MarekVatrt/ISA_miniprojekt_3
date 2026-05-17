from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

#najdeme prvy numericky stlpec v df podla zadaneho listu
def _pick_numeric_series(
    df: pd.DataFrame, candidates: list[str]
) -> Tuple[Optional[pd.Series], Optional[str]]:
    for name in candidates:
        if name in df.columns:
            try:
                s = pd.to_numeric(df[name], errors="coerce")
                if s.notna().any():
                    return s, name
            except Exception:
                continue
    return None, None


#funkcia na vykreslovanie distribucii
def plot_similarity_distribution(recommendations_df: pd.DataFrame) -> go.Figure:

    df = recommendations_df.copy() if recommendations_df is not None else pd.DataFrame()
    series, name = _pick_numeric_series(
        df, ["similarity", "hybrid_score", "top_rated_score", "average_rating"]
    )

    if series is None or series.dropna().size < 2:
        #ak nie je dost informacii na vykreslenie grafu - prazdny graf (fallback)
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="Not enough numeric score data to display a histogram",
            showarrow=False,
            font=dict(size=14),
            xref="paper",
            yref="paper",
        )
        fig.update_layout(template="plotly_white", xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
        return fig

    #ak je dost dat vykreslime histogramy
    plot_df = pd.DataFrame({"score": series.dropna()})
    title_map = {
        "similarity": "Distribution of Cosine Similarity",
        "hybrid_score": "Distribution of Hybrid Scores",
        "top_rated_score": "Distribution of Top-rated Scores",
        "average_rating": "Distribution of Average Ratings",
    }
    fig = px.histogram(
        plot_df,
        x="score",
        nbins=20,
        title=title_map.get(name, "Score distribution"),
        labels={"score": title_map.get(name, "Score")},
        color_discrete_sequence=["#2b8cbe"],
    )
    mean = float(plot_df["score"].mean())
    #vykreslime aj priemer
    fig.add_vline(x=mean, line_dash="dash", line_color="#d62728", annotation_text=f"mean={mean:.2f}", annotation_position="top left")
    fig.update_layout(showlegend=False, template="plotly_white")
    return fig

#scatterplot pre rating vs similarity
def plot_rating_vs_similarity(recommendations_df: pd.DataFrame) -> go.Figure:

    df = recommendations_df.copy() if recommendations_df is not None else pd.DataFrame()
    x_series, x_name = _pick_numeric_series(
        df, ["similarity", "hybrid_score", "top_rated_score"]
    )
    y_series = None
    if "average_rating" in df.columns:
        y_series = pd.to_numeric(df["average_rating"], errors="coerce")

    #ak mame x aj y pouzijeme scatter plot
    if x_series is not None and y_series is not None and x_series.dropna().size >= 2 and y_series.dropna().size >= 2:
        plot_df = df.loc[x_series.index].copy()
        #priprava dat
        plot_df[x_name] = pd.to_numeric(plot_df[x_name], errors="coerce")
        plot_df["average_rating"] = pd.to_numeric(plot_df["average_rating"], errors="coerce")
        #nastavime velkost "bubliny"
        size_col: Optional[str] = "ratings_count" if "ratings_count" in plot_df.columns else None
        fig = px.scatter(
            plot_df,
            x=x_name,
            y="average_rating",
            size=size_col,
            hover_data=[c for c in ["title", "authors"] if c in plot_df.columns],
            title="Average Rating vs Content Similarity",
            color="average_rating",
            color_continuous_scale="Viridis",
        )

        #vykreslime regresnu krivku
        xvals = plot_df[x_name].dropna().to_numpy()
        yvals = plot_df.loc[plot_df[x_name].notna(), "average_rating"].to_numpy()
        if xvals.size >= 2 and yvals.size >= 2:
            try:
                m, b = np.polyfit(xvals, yvals, 1)
                line_x = np.linspace(xvals.min(), xvals.max(), 100)
                line_y = m * line_x + b
                fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color="#444444", dash="dash"), name="trend"))
            except Exception:
                #ignorujeme regression errory
                pass
        fig.update_layout(template="plotly_white")
        return fig

    #ak sa nepodari vypocitat similarity, fallback je vypisanie top rated books cez bar chart
    if "average_rating" in df.columns and "title" in df.columns:
        plot_df = df[["title", "average_rating"]].dropna().copy()
        if plot_df.empty:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="Not enough data to plot ratings", showarrow=False, xref="paper", yref="paper")
            fig.update_layout(template="plotly_white")
            return fig
        #top N knih (teraz 20)
        plot_df = plot_df.sort_values("average_rating", ascending=False).head(20)
        fig = px.bar(
            plot_df,
            x="average_rating",
            y="title",
            orientation="h",
            title="Average Ratings (top items)",
            labels={"average_rating": "Average Rating", "title": "Title"},
            color_discrete_sequence=["#2b8cbe"],
        )
        fig.update_layout(template="plotly_white")
        return fig

    #ak v datasete nie je vobec nic (ani nazvy columns ani similarity..), vypiseme prazdnu fig
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text="No numeric data available for this plot", showarrow=False, xref="paper", yref="paper")
    fig.update_layout(template="plotly_white")
    return fig
