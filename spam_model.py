import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle



def local_css(spam_css):
    with open(spam_css) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("spam_css.css")  # Load the CSS file

# Rest of your app code...


# Download NLTK data
nltk.download('stopwords')

# Title and description
st.title("📧 Spam Message Detector")
st.write("""
This app predicts whether a text message is **spam** or **ham** (not spam) using Machine Learning.
""")

# Sidebar with options
st.sidebar.header("About")
st.sidebar.info("""
This model uses:
- Naive Bayes classifier
- TF-IDF for text vectorization
- NLTK for text preprocessing
""")

# Load or train model (simplified version)
@st.cache_resource
def load_model_and_vectorizer():
    # Load your trained model and vectorizer here
    # For this example, we'll create a simple one
    
    # Sample data (in a real app, load your trained model)
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    
    # Text preprocessing
    stemmer = PorterStemmer()
    def clean_text(text):
        text = ''.join([char for char in text if char not in string.punctuation])
        text = text.lower()
        text = ' '.join([stemmer.stem(word) for word in text.split() 
                        if word not in stopwords.words('english')])
        return text
    
    data['cleaned_message'] = data['message'].apply(clean_text)
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['cleaned_message']).toarray()
    y = data['label'].map({'ham': 0, 'spam': 1})
    
    # Train model
    model = MultinomialNB()
    model.fit(X, y)
    
    return model, tfidf

model, tfidf = load_model_and_vectorizer()

# Prediction function
def predict_spam(text):
    # Clean the text
    stemmer = PorterStemmer()
    def clean_text(text):
        text = ''.join([char for char in text if char not in string.punctuation])
        text = text.lower()
        text = ' '.join([stemmer.stem(word) for word in text.split() 
                        if word not in stopwords.words('english')])
        return text
    
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(vector)
    probability = model.predict_proba(vector)
    
    return prediction[0], probability[0]

# User input
user_input = st.text_area("Enter a message to check if it's spam:", 
                         "Congratulations! You've won a $1000 gift card!")

if st.button('Predict'):
    if user_input:
        prediction, probability = predict_spam(user_input)
        
        st.subheader("Result")
        if prediction == 1:
            st.error(f"🚨 This is **SPAM** (confidence: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ This is **NOT SPAM** (confidence: {probability[0]*100:.2f}%)")
        
        # Show probability breakdown
        st.write("\n")
        st.write("**Prediction Confidence:**")
        prob_df = pd.DataFrame({
            'Class': ['Not Spam (Ham)', 'Spam'],
            'Probability': [probability[0], probability[1]]
        })
        st.bar_chart(prob_df.set_index('Class'))
    else:
        st.warning("Please enter a message to analyze.")

# Sample messages to try
st.subheader("Try these examples:")
sample_spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
sample_ham = "Hey, are we still meeting for lunch tomorrow?"

if st.button("Load Spam Example"):
    st.session_state.user_input = sample_spam

if st.button("Load Ham Example"):
    st.session_state.user_input = sample_ham

# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-learn, and Streamlit")