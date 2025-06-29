# sms_spam_classifier_app.py
import streamlit as st
import joblib
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

model = joblib.load("C:\\Users\\ASUS\\OneDrive\\Documents\\LLM\\sms_spam_classifier_svm.pkl")
print("✅ Model loaded successfully!")
# -----------------------------
# Load or Train Model (Fallback)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("C:\\Users\\ASUS\\OneDrive\\Documents\\LLM\\sms_spam_classifier_svm.pkl")
    except:
        # Train a dummy fallback model if file not found
        df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['message'] = df['message'].apply(clean_text)

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svm', LinearSVC())
        ])
        pipe.fit(df['message'], df['label'])
        joblib.dump(pipe, "sms_spam_classifier_svm.pkl")
        return pipe

# -----------------------------
# Clean text helper
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📲 SMS Spam Classifier")
st.markdown("Enter an SMS message below to check if it's **Spam** or **Ham**.")

# Load model
model = load_model()

# Text input
user_input = st.text_area("📩 Enter SMS message here:", height=150)

# Predict
if st.button("🔍 Classify Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        label = "🚫 Spam" if prediction == 1 else "✅ Ham"
        st.markdown(f"### Result: {label}")

# Optionally download model
if st.button("💾 Save Trained Model"):
    joblib.dump(model, "sms_spam_classifier_svm.pkl")
    st.success("Model saved as `sms_spam_classifier_svm.pkl` ✅")
