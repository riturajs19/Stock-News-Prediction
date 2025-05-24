import streamlit as st
import pickle
import pandas as pd
import numpy as np
import feedparser
from collections import Counter

# Load model and vectorizer
model = pickle.load(open('rf_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Stock News Sentiment Analyzer", layout="wide")

st.title("📰 Stock News Sentiment Analyzer")
st.markdown("Predict sentiment (Positive/Negative) from news headlines using a trained Random Forest model.")

# --------------------------
# 🔹 Single Headline Input
# --------------------------
st.header("📝 Single Headline Prediction")

user_input = st.text_area("Enter a single news headline:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a headline.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        proba = model.predict_proba(transformed_input)[0]

        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        confidence = np.max(proba) * 100

        st.success(f"**Predicted Sentiment: {sentiment}**")
        st.info(f"Model Confidence: **{confidence:.2f}%**")

        if confidence < 60:
            st.warning("⚠️ Low confidence – the prediction might not be reliable.")

# --------------------------
# 🔹 Batch Headline Input
# --------------------------
st.header("📄 Batch Prediction")

multi_input = st.text_area("Enter multiple headlines (one per line):", height=200)

if st.button("Predict Batch"):
    lines = [line.strip() for line in multi_input.split("\n") if line.strip()]
    if lines:
        transformed = vectorizer.transform(lines)
        predictions = model.predict(transformed)

        results_df = pd.DataFrame({
            "Headline": lines,
            "Sentiment": ["Positive" if p == 1 else "Negative" for p in predictions]
        })

        sentiment_counts = Counter(results_df["Sentiment"])

        st.subheader("🔍 Results")
        st.dataframe(results_df)

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(results_df["Sentiment"].value_counts())

        if len(set(predictions)) == 1:
            st.warning("⚠️ Warning: All predictions are the same. Model may be biased or overfitting.")

# --------------------------
# 🌐 Live RSS Feed Input
# --------------------------
st.header("🌍 Live RSS News Sentiment")

rss_url = st.text_input("Enter RSS feed URL:", "https://news.google.com/rss")

if st.button("Fetch News & Predict"):
    feed = feedparser.parse(rss_url)
    headlines = [entry.title for entry in feed.entries[:10]]

    if not headlines:
        st.warning("No headlines found in the RSS feed.")
    else:
        st.write("📌 **Fetched Headlines**:")
        for h in headlines:
            st.markdown(f"- {h}")

        transformed = vectorizer.transform(headlines)
        predictions = model.predict(transformed)

        results_df = pd.DataFrame({
            "Headline": headlines,
            "Sentiment": ["Positive" if p == 1 else "Negative" for p in predictions]
        })

        st.subheader("🔍 Prediction Results")
        st.dataframe(results_df)

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(results_df["Sentiment"].value_counts())

        if len(set(predictions)) == 1:
            st.warning("⚠️ All predicted sentiments are the same – consider retraining the model or reviewing the input format.")
