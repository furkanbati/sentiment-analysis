import streamlit as st
import joblib
import re

# load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def clean(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

st.title("🎬 Film Review Sentiment Analysis")
user_input = st.text_area("Enter a film review:")

if st.button("Analyze"):
    # Prepare the input
    clean_text = clean(user_input)
    vector = tfidf.transform([clean_text])
    
    # Make a prediction
    result = model.predict(vector)
    
    if result[0] == 'positive':
        st.success("Its a POSITIVE review 😊")
    else:
        st.error("Its a NEGATIVE review 😞")
