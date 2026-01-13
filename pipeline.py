# pipeline.py
import pandas as pd

def load_data():
    return pd.read_pickle("data/resumes_with_tfidf.pkl")

def rank_candidates(df, top_k=10):

    # SAFETY CHECK (prevents silent crashes)
    required_cols = [
        "tfidf_similarity",
        "sbert_similarity",
        "rule_score"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns before ranking: {missing}")

    # FINAL SCORE (same logic you used in notebook)
    df = df.copy()
    df["final_score"] = (
        0.2 * df["tfidf_similarity"] +
        0.5 * df["sbert_similarity"] +
        0.3 * df["rule_score"]
    )

    return df.sort_values("final_score", ascending=False).head(top_k)
