# Stock-News-Prediction
The Stock News Sentiment Analyzer is a machine learning-based application designed to predict the sentiment of stock market news headlines. Understanding market sentiment is crucial for investors and analysts to make informed decisions, as news headlines often impact stock prices significantly.This project utilizes a dataset containing daily top 25 news headlines related to stocks from 2014 to 2016, each labeled with either positive or negative sentiment. By leveraging Natural Language Processing (NLP) techniques such as Bag-of-Words vectorization and a Random Forest classifier, the model learns to classify news headlines accurately into positive or negative categories.
The project is complemented by a user-friendly interactive web application built with Streamlit, allowing users to input headlines manually, analyze multiple headlines at once, or even fetch live news via RSS feeds for real-time sentiment prediction.

## 🧠 Data Preprocessing
Headlines are cleaned and prepared for analysis.
Text is transformed into numerical features using the Bag-of-Words (BoW) approach, which counts the frequency of words to represent each headline as a vector.This transformation enables the machine learning model to process textual data efficiently.

## 🔍 Model Training and Evaluation
A Random Forest Classifier is chosen due to its robustness and ability to handle complex data patterns.
The model is trained on the vectorized training dataset.
Performance is evaluated using common classification metrics (accuracy, precision, recall, etc.), ensuring reliable sentiment predictions. After training, the model and the vectorizer are serialized using Python’s pickle module to enable quick loading during inference.

## 🌐 Streamlit Web Application
The app (app1.py) provides a clean and intuitive interface for users to interact with the model.
Single Headline Prediction:
-Users can input any stock market news headline and instantly receive a sentiment prediction.
Batch Prediction:
-Allows users to input multiple headlines separated by new lines and get bulk sentiment predictions.
Live RSS Feed Integration:
-Users can enter an RSS feed URL to fetch the latest stock-related news headlines.
-The app processes these headlines and displays their predicted sentiments along with confidence scores.
Visualizations:
-The app presents sentiment distribution charts for batch and live RSS feed predictions, providing quick insights into the market mood.

## 📝 Summary
This Stock News Sentiment Analyzer project showcases a practical application of machine learning and NLP to the financial domain. By analyzing the sentiment of stock-related news headlines, the tool assists users in understanding the market’s emotional context, which can often influence stock price movements. The integration of a Random Forest model with Bag-of-Words vectorization ensures robust and interpretable predictions. The Streamlit-based web interface makes the system accessible to both technical and non-technical users, supporting real-time sentiment analysis from multiple input sources, including live RSS news feeds.

This project highlights the power of combining data science and software engineering to deliver actionable insights in the fast-moving world of stock markets. Future improvements can include incorporating advanced NLP models like transformers, expanding dataset coverage, and integrating more granular sentiment scoring.
