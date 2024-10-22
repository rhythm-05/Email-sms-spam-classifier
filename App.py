import streamlit as st
import pickle
import string
import spacy
import os

# Define the spaCy model name
MODEL_NAME = 'en_core_web_sm'


# Ensure spaCy model is downloaded
def ensure_spacy_model(model_name=MODEL_NAME):
    try:
        spacy.load(model_name)
    except OSError:
        os.system(f"python -m spacy download {model_name}")


# Ensure the spaCy model is installed
ensure_spacy_model()

# Load spaCy's English tokenizer
nlp = spacy.load(MODEL_NAME)


def transform_text(text):
    text = text.lower()
    doc = nlp(text)  # Tokenize using spaCy
    tokens = [token.text for token in doc if token.is_alpha]  # Filter alphanumeric tokens

    # Remove stopwords and punctuation
    filtered_tokens = [token for token in tokens if
                       token not in nlp.Defaults.stop_words and token not in string.punctuation]

    # Note: spaCy uses lemmatization instead of stemming. If stemming is needed, use another library or approach.

    return " ".join(filtered_tokens)


# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app code
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
