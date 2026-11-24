
import streamlit as st
import joblib
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Increase font sizes
st.markdown("""
<style>

/* Main title */
h1 {
    font-size: 38px !important;
}

/* Normal text */
p, label, .stMarkdown {
    font-size: 20px !important;
}

/* Text area input */
.stTextArea textarea {
    font-size: 20px !important;
}

/* Text area label */
.stTextArea label {
    font-size: 21px !important;
}

/* Analyze button */
.stButton button {
    font-size: 20px !important;
    font-weight: bold;
    height: 50px;
}

/* Prediction messages */
.stAlert p {
    font-size: 21px !important;
}

/* Footer caption */
.stCaption {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# Load saved model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


svm_model, tfidf = load_models()


# Text preprocessing
stop_words = set(stopwords.words("english"))

negation_words = {
    "no", "not", "nor", "never",
    "dont", "doesnt", "didnt",
    "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "couldnt", "shouldnt",
    "cant", "cannot"
}

stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r"<.*?>", " ", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    processed_words = []

    for word in words:
        if word == "us":
            processed_words.append(word)
        else:
            processed_words.append(lemmatizer.lemmatize(word))

    return " ".join(processed_words)


# Application interface
st.title("💬 Customer Review Sentiment Analysis")

st.write(
    "Enter a customer review below to predict "
    "whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: This product is amazing and works perfectly!",
    height=150
)


if st.button("Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        cleaned_review = preprocess_text(review)
        review_tfidf = tfidf.transform([cleaned_review])

        prediction = svm_model.predict(review_tfidf)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")


st.caption(
    "Powered by Machine Learning | SVM Model"
)
