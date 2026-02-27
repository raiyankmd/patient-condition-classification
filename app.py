import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load trained pipeline
@st.cache_resource
def load_pipeline():
    with open("patient_condition_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# UI
st.title("🩺 Patient Condition Classification")
st.write("Enter a patient's drug review to classify the condition.")

user_review = st.text_area("Patient Review")

if st.button("Predict Condition"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        try:
            prediction = pipeline.predict([user_review])
            st.success(f"Predicted Condition: {prediction[0]}")
        except Exception as e:
            st.error(f"Error: {e}")
