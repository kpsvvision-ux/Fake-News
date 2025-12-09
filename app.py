import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


st.title("Fake News Detection Application")

# Function to clean text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# ======================= Upload Training Data ==========================
st.subheader("Upload Your Training Datasets")
fake_file = st.file_uploader("Upload Fake.csv", type=["csv"])
true_file = st.file_uploader("Upload True.csv", type=["csv"])


if fake_file and true_file:

    st.success("Training datasets uploaded successfully.")

    # ==================== PREPARE DATA ==========================
    @st.cache_resource
    def load_and_prepare_data(fake_df, true_df):
        fake_df["class"] = 0
        true_df["class"] = 1

        fake_df = fake_df.iloc[:-10]
        true_df = true_df.iloc[:-10]

        df = pd.concat([fake_df, true_df], axis=0)
        df = df.drop(["title", "subject", "date"], axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        df["clean_text"] = df["text"].apply(wordopt)
        return df

    # ==================== TRAIN MODEL ==========================
    @st.cache_resource
    def train_model(df):
        x = df["clean_text"]
        y = df["class"]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(x)

        LR = LogisticRegression()
        LR.fit(X, y)

        return vectorizer, LR

    fake_df = pd.read_csv(fake_file)
    true_df = pd.read_csv(true_file)

    df = load_and_prepare_data(fake_df, true_df)
    vectorizer, LR = train_model(df)

    # ==================== Option 1: Manual Text ==========================
    st.markdown("---")
    st.subheader("Option 1: Classify Manually Entered Text")

    news_input = st.text_area("Enter news text:", height=150)

    if st.button("Classify Text"):
        if news_input.strip():
            cleaned = wordopt(news_input)
            x_vec = vectorizer.transform([cleaned])
            pred = LR.predict(x_vec)[0]
            st.success(f"Prediction: {'Fake News' if pred == 0 else 'Not Fake News'}")
        else:
            st.warning("Please enter some text to classify.")

    # ==================== Option 2: Predict Random Training Records ==========================
    st.markdown("---")
    st.subheader("Option 2: Predict Random Records From Uploaded Dataset")

    sample_size = st.slider(
        "How many random records do you want to predict?",
        min_value=1,
        max_value=20,
        value=5
    )

    if st.button("Predict Random Records"):
        sample = df.sample(sample_size).copy()
        sample_vectors = vectorizer.transform(sample["clean_text"])
        sample["Prediction"] = LR.predict(sample_vectors)
        sample["Prediction"] = sample["Prediction"].apply(
            lambda x: "Fake News" if x == 0 else "Not Fake News"
        )

        st.subheader("Random Predictions:")
        st.dataframe(sample[["text", "Prediction"]])

        csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv,
            "random_predictions.csv",
            "text/csv"
        )

else:
    st.info("Please upload both Fake.csv and True.csv to train the model.")
