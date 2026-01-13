import streamlit as st
import pandas as pd
from pipeline import load_data, rank_candidates

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("📄 AI Resume Screener")
st.markdown("Hybrid ranking using **Skills + TF-IDF + SBERT**")

@st.cache_data
def load_cached_data():
    return load_data()

df = load_cached_data()

# Job role selector
job_roles = sorted(df["Category"].unique())
selected_role = st.selectbox("Select Job Role", job_roles)

filtered_df = df[df["Category"] == selected_role]

# Top-K slider
top_k = st.slider("Number of candidates", 5, 20, 10)

ranked = rank_candidates(filtered_df, top_k)

st.subheader("🏆 Top Candidates")

st.dataframe(
    ranked[
        [
            "skills",
            "rule_score",
            "tfidf_similarity",
            "sbert_similarity",
            "final_score"
        ]
    ],
    use_container_width=True
)

st.subheader("📊 Scoring Logic")
st.markdown("""
- **Rule-based match** (skills, experience, education): 30%
- **TF-IDF similarity**: 20%
- **SBERT semantic similarity**: 50%
""")

with st.expander("🔍 Preview Top Resume"):
    st.write(ranked.iloc[0]["cleaned_resume"][:1500])
