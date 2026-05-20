import os
import streamlit as st
import pandas as pd
from utils.config import CLEAN_DATA_PATH
from transformers import pipeline
from google import genai
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# Cached functions (load once, reuse)
# -------------------------------------------------
@st.cache_resource
def load_local_model():
    """
    Load the lightweight DistilBERT sentiment model.
    The model is cached so it is instantiated only once per
    Streamlit session.
    """
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def get_gemini_client():
    """Create a Gemini client (uses GEMINI_API_KEY from .env)."""
    return genai.Client()

@st.cache_data
def load_data():
    """Read the cleaned CSV that lives in the data folder."""
    return pd.read_csv(CLEAN_DATA_PATH)

# -------------------------------------------------
# Initialise resources
# -------------------------------------------------
local_analyzer = load_local_model()
gemini_client = get_gemini_client()
df = load_data()

# -------------------------------------------------
# Dashboard layout
# -------------------------------------------------
st.title("🗣️ Twitter Sentiment Analysis Pipeline")
st.markdown("---")

# ----- Left column: KPIs and sentiment chart -----
col_kpi, col_chart = st.columns([1, 2])

with col_kpi:
    st.subheader("📈 Key Metrics")
    total_tweets = len(df)
    st.metric(label="Total Tweets Analyzed", value=total_tweets)

    # Sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

# ----- Right column: Explorer / filters -----
with col_chart:
    st.subheader("🔎 Tweet Explorer")
    # Brand filter
    brand_options = ["All"] + sorted(df["brand"].dropna().unique().tolist())
    selected_brand = st.selectbox("Filter by brand / product", brand_options)

    # Text search
    search_term = st.text_input("Search tweet text")

    # Apply filters
    filtered = df.copy()
    if selected_brand != "All":
        filtered = filtered[filtered["brand"] == selected_brand]
    if search_term:
        filtered = filtered[
            filtered["text"].str.contains(search_term, case=False, na=False)
        ]

    # Show filtered table
    st.dataframe(
        filtered[["text", "brand", "sentiment"]],
        use_container_width=True
    )

st.markdown("---")

# -------------------------------------------------
# Live IA section (Hybrid inference)
# -------------------------------------------------
st.subheader("🤖 Live Hybrid Inference (DistilBERT + Gemini)")
st.markdown(
    "Type a tweet (English) below and press **Enter**. "
    "If the local model is uncertain or the tweet contains "
    "potentially ambiguous keywords, the request will be routed to Gemini."
)

tweet = st.text_input(
    "Enter your tweet here:",
    placeholder="e.g. can not wait for iPad 2 also they should sell them down at SXSW"
)

if tweet:
    # 1️⃣ Local inference
    local_res = local_analyzer([tweet])[0]
    local_label = local_res["label"]
    local_score = local_res["score"]

    # 2️⃣ Routing decision
    ambiguous_keywords = ["wait", "bankrupt", "but", "although", "despite"]
    route_to_gemini = (
        local_score < 0.95
        or any(word in tweet.lower() for word in ambiguous_keywords)
    )

    # 3️⃣ UI layout for results
    col_local, col_remote = st.columns(2)

    with col_local:
        st.info("💻 Local model (DistilBERT)")
        st.write(f"**Prediction:** {local_label}")
        st.write(f"**Confidence:** {local_score:.2%}")

    with col_remote:
        if route_to_gemini:
            st.warning("⚠️ Routed to Gemini (semantic complexity)")
            with st.spinner("Calling Gemini API…"):
                prompt = (
                    "Analyse the sentiment of this tweet. Respond ONLY with "
                    "POSITIVE, NEGATIVE, or NEUTRAL.\n"
                    f"Tweet: '{tweet}'"
                )
                try:
                    resp = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    gemini_result = resp.text.strip().upper()
                    st.success(f"**Gemini result:** {gemini_result} ✅")
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
        else:
            st.success("🟢 Processed locally (high confidence)")
            st.success(f"**Final result:** {local_label} ✅")
