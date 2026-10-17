"""
Streamlit web interface.
Takes an email text from the user and classifies it as spam or normal.
"""

import random
import pandas as pd
import streamlit as st
from predict import SpamPredictor

st.set_page_config(
    page_title="Spam Email Detection System",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Spam Email Detection System")
st.markdown("Enter an email text and let the AI analyze it.")
st.divider()


@st.cache_resource
def load_predictor():
    """Loads the model once and caches it."""
    return SpamPredictor()


@st.cache_data
def load_samples():
    """Loads all messages from the CSV."""
    df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df["text"].dropna().tolist()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Initialize session state
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# Sample data buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎲 Random Sample", use_container_width=True):
        samples = load_samples()
        st.session_state["text_input"] = random.choice(samples)

with col2:
    if st.button("🚨 Spam Sample", use_container_width=True):
        df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]
        st.session_state["text_input"] = random.choice(df[df["label"] == "spam"]["text"].dropna().tolist())

with col3:
    if st.button("✅ Normal Sample", use_container_width=True):
        df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]
        st.session_state["text_input"] = random.choice(df[df["label"] == "ham"]["text"].dropna().tolist())

# Text area
text_input = st.text_area(
    "Paste the email text here:",
    value=st.session_state["text_input"],
    height=200,
    placeholder="Enter the email content...",
)

st.divider()

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = predictor.predict(text_input)

        if result["is_spam"]:
            st.error(f"### Result: {result['label']}")
            st.metric("Confidence", f"{result['confidence']}%")
            st.markdown(
                "> The AI classified this email as **spam**. "
                "Be careful and do not click suspicious links."
            )
        else:
            st.success(f"### Result: {result['label']}")
            st.metric("Confidence", f"{result['confidence']}%")
            st.markdown(
                "> The AI classified this email as **normal**."
            )

st.divider()
st.caption("AI-Powered Spam Email Detection System | TF-IDF + Logistic Regression")
