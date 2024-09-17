import streamlit as st
import pickle
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy's English tokenizer
nlp = spacy.load('en_core_web_sm')


def transform_text(text):
    text = text.lower()
    doc = nlp(text)  # Tokenize using spaCy
    tokens = [token.text for token in doc if token.is_alpha]  # Filter alphanumeric tokens

    # Remove stopwords and punctuation
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation]

    # Stem tokens
    # Note: spaCy doesn't provide stemming; it uses lemmatization. If you specifically need stemming, you can use another library.
    # For now, we skip stemming for simplicity.

    return " ".join(filtered_tokens)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
