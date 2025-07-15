import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_news(news_title):
    processed_text = news_title.lower()  # Simplified preprocessing
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    return "Real News ✅" if prediction[0] == 1 else "Fake News ❌"

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news headline to check if it's real or fake")

news_input = st.text_input("News Headline:", "")

if st.button("Predict"):
    if news_input:
        result = predict_news(news_input)
        st.success(result)
    else:
        st.warning("Please enter a news headline!")
