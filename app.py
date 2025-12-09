
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
from sklearn.metrics import classification_report, accuracy_score


st.title("Fake News Detection Application")

# Function to clean the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load and preprocess data (cached to run only once)
@st.cache_resource
def load_data():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["class"] = 0
    df_true["class"] = 1

    # Removing last 10 rows for manual testing (as done in notebook)
    # This part is omitted for the main training data in the app,
    # as the app focuses on predicting new input.
    # The original notebook removed rows *before* merging.
    # To mimic the original notebook's training data, we should keep this.
    # However, for a practical app, we typically train on full data.
    # For direct conversion, I'll remove them similar to the notebook for consistency.
    df_fake = df_fake.iloc[:-10]
    df_true = df_true.iloc[:-10]

    df_merge = pd.concat([df_fake, df_true], axis=0)
    df_merge = df_merge.drop(["title", "subject", "date"], axis=1)
    df_merge = df_merge.sample(frac=1).reset_index(drop=True)

    df_merge["text"] = df_merge["text"].apply(wordopt)

    x = df_merge["text"]
    y = df_merge["class"]

    return x, y

# Train models (cached to run only once)
@st.cache_resource
def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Logistic Regression
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)

    # Decision Tree Classification
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)

    # Gradient Boosting Classifier
    GBC = GradientBoostingClassifier(random_state=0, n_estimators=50, max_depth=2, subsample=0.8, max_features='sqrt')
    GBC.fit(xv_train, y_train)

    # Random Forest Classifier
    RFC = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10, n_jobs=-1)
    RFC.fit(xv_train, y_train)

    return vectorization, LR, DT, GBC, RFC, xv_test, y_test

# Helper function to convert prediction to label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Main application logic
x, y = load_data()
vectorization, LR, DT, GBC, RFC, xv_test, y_test = train_models(x, y)

st.subheader("Model Performance on Test Set (Accuracy):")
st.write(f"Logistic Regression: {LR.score(xv_test, y_test):.4f}")
st.write(f"Decision Tree: {DT.score(xv_test, y_test):.4f}")
st.write(f"Gradient Boosting: {GBC.score(xv_test, y_test):.4f}")
st.write(f"Random Forest: {RFC.score(xv_test, y_test):.4f}")

st.subheader("Enter a news article to classify:")
news_input = st.text_area("", "", height=200)

if st.button("Classify News"):
    if news_input:
        # Preprocess the input news
        processed_news = wordopt(news_input)
        testing_news_df = pd.DataFrame({"text": [processed_news]})

        # Transform using the trained vectorizer
        new_xv_test = vectorization.transform(testing_news_df["text"])

        # Make predictions
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)

        st.subheader("Prediction Results:")
        st.write(f"**Logistic Regression Prediction:** {output_lable(pred_LR[0])}")
        st.write(f"**Decision Tree Prediction:** {output_lable(pred_DT[0])}")
        st.write(f"**Gradient Boosting Prediction:** {output_lable(pred_GBC[0])}")
        st.write(f"**Random Forest Prediction:** {output_lable(pred_RFC[0])}")
    else:
        st.warning("Please enter some text to classify.")
